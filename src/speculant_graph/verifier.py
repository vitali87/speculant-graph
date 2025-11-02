import torch
import random
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from loguru import logger

from speculant_graph.draft_generator import DraftGenerator
from speculant_graph.config import VerifierConfig, GenerationConfig, DraftConfig
from speculant_graph.download_utils import configure_download_mode


class GenerationResult(BaseModel):
    text: str
    token_ids: list[int]
    acceptance_rate: float
    num_accepted: int
    num_rejected: int
    total_tokens: int
    position_acceptance_counts: dict[int, int]  # Maps position to acceptance count
    position_proposal_counts: dict[int, int]  # Maps position to proposal count


class SpeculativeDecoder:
    def __init__(
        self,
        graph_path: str,
        verifier_config: VerifierConfig,
        draft_config: DraftConfig,
    ):
        self.verifier_config = verifier_config
        self.draft_config = draft_config

        configure_download_mode(verifier_config.download_mode)

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = (
            dtype_map.get(verifier_config.torch_dtype)
            if verifier_config.torch_dtype
            else None
        )

        logger.info(f"Loading verifier model: {verifier_config.model_name}")
        logger.info(
            f"Memory config: dtype={verifier_config.torch_dtype}, "
            f"device_map={verifier_config.device_map}, "
            f"low_cpu_mem_usage={verifier_config.low_cpu_mem_usage}"
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            verifier_config.model_name,
            token=verifier_config.hf_token,
            dtype=torch_dtype,
            device_map=verifier_config.device_map,
            low_cpu_mem_usage=verifier_config.low_cpu_mem_usage,
        )
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = True
        self.tokenizer = AutoTokenizer.from_pretrained(
            verifier_config.model_name, token=verifier_config.hf_token
        )

        if verifier_config.device_map:
            self.device = next(self.model.parameters()).device
            logger.info(f"Model loaded with device_map: {verifier_config.device_map}")
        else:
            self.device = verifier_config.device or (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self.model.to(self.device)
            logger.info(f"Model loaded on device: {self.device}")

        logger.info(f"Loading n-gram graph from: {graph_path}")
        self.draft_generator = DraftGenerator.from_file(
            graph_path,
            verifier_config.model_name,
            verifier_config.hf_token,
            verifier_config.download_mode,
            draft_config,
            tokenizer=self.tokenizer,
        )
        logger.info("N-gram graph loaded successfully")
        self._reset_verifier_cache()

    def generate(
        self, prompt: str, generation_config: GenerationConfig
    ) -> GenerationResult:
        if generation_config.seed is not None:
            random.seed(generation_config.seed)
            torch.manual_seed(generation_config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(generation_config.seed)
            logger.debug(f"Set random seed to {generation_config.seed}")

        prompt_preview = prompt if len(prompt) <= 100 else f"{prompt[:100]}..."
        logger.info(f"Generating with prompt: '{prompt_preview}'")
        logger.info(
            f"Config: max_tokens={generation_config.max_tokens}, "
            f"temperature={generation_config.temperature}, "
            f"seed={generation_config.seed}"
        )

        input_ids = self._prepare_input_ids(prompt)
        self._reset_verifier_cache()
        self._prime_verifier_state(input_ids)

        generated_tokens = input_ids[0].tolist()
        prompt_length = len(input_ids[0])

        num_draft_accepted = 0
        num_draft_rejected = 0
        num_verifier_generated = 0
        position_acceptance_counts: dict[
            int, int
        ] = {}  # Track acceptances per position
        position_proposal_counts: dict[int, int] = {}  # Track proposals per position

        while len(generated_tokens) - prompt_length < generation_config.max_tokens:
            current_text = self.tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            )

            tokens_remaining = generation_config.max_tokens - (
                len(generated_tokens) - prompt_length
            )
            draft_k = min(self.draft_config.k, tokens_remaining)

            draft_result = self.draft_generator.generate(
                prompt=current_text, k=draft_k, strategy=self.draft_config.strategy
            )

            if len(draft_result.token_ids) == 0:
                logger.debug(
                    "Draft generator returned empty sequence, generating from verifier"
                )
                verifier_count, new_tokens = self._generate_from_verifier(
                    count=1, temperature=generation_config.temperature
                )
                num_verifier_generated += verifier_count
                generated_tokens.extend(new_tokens)
                continue

            (
                accepted_count,
                rejected_count,
                accepted_tokens,
                has_correction,
                accepted_positions,
            ) = self._verify_draft(
                generated_tokens,
                draft_result.token_ids,
                draft_result.token_probs,
                draft_result.matched_contexts,
                draft_result.successors,
                draft_result.successor_weights,
                draft_result.strategy,
                temperature=generation_config.temperature,
            )

            num_draft_accepted += accepted_count
            num_draft_rejected += rejected_count

            for i in range(len(draft_result.token_ids)):
                position_proposal_counts[i] = position_proposal_counts.get(i, 0) + 1
            for pos in accepted_positions:
                position_acceptance_counts[pos] = (
                    position_acceptance_counts.get(pos, 0) + 1
                )

            if accepted_count > 0:
                accepted_draft = draft_result.token_ids[:accepted_count]
                accepted_text = self.tokenizer.decode(
                    accepted_draft, skip_special_tokens=True
                )
                logger.info(
                    f"✓ Accepted {accepted_count}/{len(draft_result.token_ids)} draft tokens: '{accepted_text}'"
                )

            if has_correction:
                corrected_token = accepted_tokens[-1]
                corrected_text = self.tokenizer.decode(
                    [corrected_token], skip_special_tokens=True
                )
                rejected_token = draft_result.token_ids[accepted_count]
                rejected_text = self.tokenizer.decode(
                    [rejected_token], skip_special_tokens=True
                )
                logger.info(
                    f"✗ Rejected draft token '{rejected_text}' (ID: {rejected_token})"
                )
                logger.info(
                    f"→ Sampled correction '{corrected_text}' (ID: {corrected_token}) from adjusted distribution"
                )
            elif accepted_count == 0:
                rejected_text = self.tokenizer.decode(
                    draft_result.token_ids, skip_special_tokens=True
                )
                logger.info(
                    f"✗ Rejected all {len(draft_result.token_ids)} draft tokens: '{rejected_text}'"
                )

            generated_tokens.extend(accepted_tokens)

            if accepted_count == 0 and not has_correction:
                logger.info("→ Generating fallback token from verifier model...")
                verifier_count, fallback_tokens = self._generate_from_verifier(
                    count=1, temperature=generation_config.temperature
                )
                fallback_text = self.tokenizer.decode(
                    fallback_tokens, skip_special_tokens=True
                )
                fallback_ids_str = ", ".join(str(t) for t in fallback_tokens)
                logger.info(
                    f"→ Verifier sampled fallback token: '{fallback_text}' (ID: {fallback_ids_str})"
                )
                num_verifier_generated += verifier_count
                generated_tokens.extend(fallback_tokens)

        final_text = self.tokenizer.decode(
            generated_tokens[prompt_length:], skip_special_tokens=True
        )
        total_draft_proposed = num_draft_accepted + num_draft_rejected
        acceptance_rate = (
            num_draft_accepted / total_draft_proposed
            if total_draft_proposed > 0
            else 0.0
        )

        logger.info(
            f"Generation complete: {len(generated_tokens) - prompt_length} tokens generated"
        )
        logger.info(
            f"Draft acceptance rate: {acceptance_rate:.2%} ({num_draft_accepted}/{total_draft_proposed})"
        )
        logger.info(
            f"Breakdown: {num_draft_accepted} draft accepted, {num_draft_rejected} draft rejected, {num_verifier_generated} verifier generated"
        )

        if position_proposal_counts:
            logger.info("Position-based acceptance statistics:")
            max_position = max(position_proposal_counts.keys())
            for pos in range(max_position + 1):
                proposals = position_proposal_counts.get(pos, 0)
                acceptances = position_acceptance_counts.get(pos, 0)
                if proposals > 0:
                    pos_rate = acceptances / proposals
                    logger.info(
                        f"  Position {pos}: {acceptances}/{proposals} ({pos_rate:.2%})"
                    )

        return GenerationResult(
            text=final_text,
            token_ids=generated_tokens[prompt_length:],
            acceptance_rate=acceptance_rate,
            num_accepted=num_draft_accepted,
            num_rejected=num_draft_rejected,
            total_tokens=len(generated_tokens) - prompt_length,
            position_acceptance_counts=position_acceptance_counts,
            position_proposal_counts=position_proposal_counts,
        )

    def _prepare_input_ids(self, prompt: str) -> torch.Tensor:
        enc = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        if len(input_ids[0]) == 0:
            starter_token = None
            if self.tokenizer.bos_token_id is not None:
                starter_token = self.tokenizer.bos_token_id
            elif self.tokenizer.eos_token_id is not None:
                starter_token = self.tokenizer.eos_token_id
            elif self.tokenizer.pad_token_id is not None:
                starter_token = self.tokenizer.pad_token_id
            else:
                starter_token = self.draft_generator.get_most_frequent_token()
            input_ids = torch.tensor([[starter_token]], device=self.device)
        return input_ids

    def generate_stream(self, prompt: str, generation_config: GenerationConfig):
        if generation_config.seed is not None:
            random.seed(generation_config.seed)
            torch.manual_seed(generation_config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(generation_config.seed)

        input_ids = self._prepare_input_ids(prompt)
        self._reset_verifier_cache()
        self._prime_verifier_state(input_ids)
        generated_tokens = input_ids[0].tolist()
        prompt_length = len(input_ids[0])
        num_draft_accepted = 0
        num_draft_rejected = 0
        num_verifier_generated = 0
        position_acceptance_counts: dict[int, int] = {}
        position_proposal_counts: dict[int, int] = {}

        while len(generated_tokens) - prompt_length < generation_config.max_tokens:
            current_text = self.tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            )
            tokens_remaining = generation_config.max_tokens - (
                len(generated_tokens) - prompt_length
            )
            draft_k = min(self.draft_config.k, tokens_remaining)
            draft_result = self.draft_generator.generate(
                prompt=current_text, k=draft_k, strategy=self.draft_config.strategy
            )

            if len(draft_result.token_ids) == 0:
                verifier_count, new_tokens = self._generate_from_verifier(
                    count=1, temperature=generation_config.temperature
                )
                num_verifier_generated += verifier_count
                for token in new_tokens:
                    generated_tokens.append(token)
                    yield (
                        self.tokenizer.decode([token], skip_special_tokens=True),
                        generated_tokens[prompt_length:],
                    )
                continue

            (
                accepted_count,
                rejected_count,
                accepted_tokens,
                has_correction,
                accepted_positions,
            ) = self._verify_draft(
                generated_tokens,
                draft_result.token_ids,
                draft_result.token_probs,
                draft_result.matched_contexts,
                draft_result.successors,
                draft_result.successor_weights,
                draft_result.strategy,
                temperature=generation_config.temperature,
            )

            num_draft_accepted += accepted_count
            num_draft_rejected += rejected_count
            for i in range(len(draft_result.token_ids)):
                position_proposal_counts[i] = position_proposal_counts.get(i, 0) + 1
            for pos in accepted_positions:
                position_acceptance_counts[pos] = (
                    position_acceptance_counts.get(pos, 0) + 1
                )

            for token in accepted_tokens:
                generated_tokens.append(token)
                yield (
                    self.tokenizer.decode([token], skip_special_tokens=True),
                    generated_tokens[prompt_length:],
                )

            if accepted_count == 0 and not has_correction:
                verifier_count, fallback_tokens = self._generate_from_verifier(
                    count=1, temperature=generation_config.temperature
                )
                num_verifier_generated += verifier_count
                for token in fallback_tokens:
                    generated_tokens.append(token)
                    yield (
                        self.tokenizer.decode([token], skip_special_tokens=True),
                        generated_tokens[prompt_length:],
                    )

        final_text = self.tokenizer.decode(
            generated_tokens[prompt_length:], skip_special_tokens=True
        )
        total_draft_proposed = num_draft_accepted + num_draft_rejected
        acceptance_rate = (
            num_draft_accepted / total_draft_proposed
            if total_draft_proposed > 0
            else 0.0
        )

        result = GenerationResult(
            text=final_text,
            token_ids=generated_tokens[prompt_length:],
            acceptance_rate=acceptance_rate,
            num_accepted=num_draft_accepted,
            num_rejected=num_draft_rejected,
            total_tokens=len(generated_tokens) - prompt_length,
            position_acceptance_counts=position_acceptance_counts,
            position_proposal_counts=position_proposal_counts,
        )
        yield result

    def _reset_verifier_cache(self) -> None:
        self.past_key_values = None
        self.attention_mask = None
        self.last_logits = None

    def _prime_verifier_state(self, input_ids: torch.Tensor) -> None:
        attention_mask = torch.ones(
            (1, input_ids.shape[1]), device=self.device, dtype=torch.long
        )
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
        self.past_key_values = outputs.past_key_values
        self.attention_mask = attention_mask
        self.last_logits = outputs.logits[:, -1, :]

    def _append_token(self, token_id: int) -> None:
        if self.attention_mask is None or self.past_key_values is None:
            raise RuntimeError("Verifier cache not initialized")

        token_tensor = torch.tensor([[token_id]], device=self.device)
        new_attention_mask = torch.cat(
            [
                self.attention_mask,
                torch.ones((1, 1), device=self.device, dtype=torch.long),
            ],
            dim=1,
        )

        with torch.no_grad():
            outputs = self.model(
                input_ids=token_tensor,
                attention_mask=new_attention_mask,
                past_key_values=self.past_key_values,
                use_cache=True,
            )

        self.past_key_values = outputs.past_key_values
        self.attention_mask = new_attention_mask
        self.last_logits = outputs.logits[:, -1, :]

    def _next_token_distribution(self, temperature: float) -> torch.Tensor:
        if self.last_logits is None:
            raise RuntimeError("Verifier cache not initialized")
        logits = self.last_logits.squeeze(0) / temperature
        return torch.softmax(logits, dim=-1)

    def _verify_draft(
        self,
        context_tokens: list[int],
        draft_tokens: list[int],
        draft_probs: list[float],
        matched_contexts: list[tuple],
        successors: list[list[int]],
        successor_weights: list[list[float]],
        strategy: str,
        temperature: float,
    ) -> tuple[int, int, list[int], bool, list[int]]:
        if self.last_logits is None:
            raise RuntimeError("Verifier cache not initialized")

        accepted_tokens = []
        accepted_positions = []  # Track which positions were accepted
        num_actually_accepted = 0
        has_correction = False

        for i, draft_token in enumerate(draft_tokens):
            target_probs = self._next_token_distribution(temperature)
            target_prob = target_probs[draft_token].item()

            if strategy == "greedy":
                acceptance_prob = target_prob
            else:
                draft_prob = draft_probs[i]
                if draft_prob <= 0.0:
                    acceptance_prob = 1.0
                else:
                    acceptance_prob = min(1.0, target_prob / draft_prob)

            if random.random() < acceptance_prob:
                accepted_tokens.append(draft_token)
                accepted_positions.append(i)  # Record the position
                num_actually_accepted += 1
                self._append_token(draft_token)
                continue

            if strategy == "greedy":
                residual = target_probs.clone()
                residual[draft_token] = 0.0
            else:
                succ = successors[i]
                q = torch.zeros_like(target_probs)
                if succ:
                    q_weights = torch.tensor(
                        successor_weights[i],
                        device=target_probs.device,
                        dtype=target_probs.dtype,
                    )
                    q[succ] = q_weights
                residual = torch.clamp(target_probs - q, min=0.0)
                if residual.sum().item() == 0.0:
                    residual = target_probs.clone()
                    residual[draft_token] = 0.0

            residual_sum = residual.sum().item()
            if residual_sum > 0.0:
                residual = residual / residual_sum
                corrected_token = torch.multinomial(residual, num_samples=1).item()
                accepted_tokens.append(corrected_token)
                has_correction = True
                self._append_token(corrected_token)
            break

        rejected_count = len(draft_tokens) - num_actually_accepted
        return (
            num_actually_accepted,
            rejected_count,
            accepted_tokens,
            has_correction,
            accepted_positions,
        )

    def _generate_from_verifier(
        self, count: int, temperature: float
    ) -> tuple[int, list[int]]:
        if self.last_logits is None:
            raise RuntimeError("Verifier cache not initialized")

        generated = []

        for _ in range(count):
            probs = self._next_token_distribution(temperature)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)
            self._append_token(next_token)

        return len(generated), generated

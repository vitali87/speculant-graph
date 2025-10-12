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

        logger.info(f"Loading verifier model: {verifier_config.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            verifier_config.model_name, token=verifier_config.hf_token
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            verifier_config.model_name, token=verifier_config.hf_token
        )

        self.device = verifier_config.device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        logger.info(f"Model loaded on device: {self.device}")

        logger.info(f"Loading knowledge graph from: {graph_path}")
        self.draft_generator = DraftGenerator.from_file(
            graph_path,
            verifier_config.model_name,
            verifier_config.hf_token,
            verifier_config.download_mode,
        )
        logger.info("Knowledge graph loaded successfully")

    def generate(
        self, prompt: str, generation_config: GenerationConfig
    ) -> GenerationResult:
        prompt_preview = prompt if len(prompt) <= 100 else f"{prompt[:100]}..."
        logger.info(f"Generating with prompt: '{prompt_preview}'")
        logger.info(
            f"Config: max_tokens={generation_config.max_tokens}, "
            f"temperature={generation_config.temperature}"
        )

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        generated_tokens = input_ids[0].tolist()

        num_draft_accepted = 0
        num_draft_rejected = 0
        num_verifier_generated = 0

        while len(generated_tokens) - len(input_ids[0]) < generation_config.max_tokens:
            current_text = self.tokenizer.decode(generated_tokens)

            tokens_remaining = generation_config.max_tokens - (
                len(generated_tokens) - len(input_ids[0])
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
                    generated_tokens, count=1, temperature=generation_config.temperature
                )
                num_verifier_generated += verifier_count
                generated_tokens.extend(new_tokens)
                continue

            accepted_count, rejected_count, accepted_tokens, has_correction = (
                self._verify_draft(
                    generated_tokens,
                    draft_result.token_ids,
                    draft_result.token_probs,
                    draft_result.matched_contexts,
                    draft_result.successors,
                    draft_result.successor_weights,
                    draft_result.strategy,
                    temperature=generation_config.temperature,
                )
            )

            num_draft_accepted += accepted_count
            num_draft_rejected += rejected_count

            # Log detailed results
            if accepted_count > 0:
                accepted_draft = draft_result.token_ids[:accepted_count]
                accepted_text = self.tokenizer.decode(accepted_draft)
                logger.info(
                    f"✓ Accepted {accepted_count}/{len(draft_result.token_ids)} draft tokens: '{accepted_text}'"
                )

            if has_correction:
                corrected_token = accepted_tokens[-1]
                corrected_text = self.tokenizer.decode([corrected_token])
                rejected_token = draft_result.token_ids[accepted_count]
                rejected_text = self.tokenizer.decode([rejected_token])
                logger.info(
                    f"✗ Rejected draft token '{rejected_text}' (ID: {rejected_token})"
                )
                logger.info(
                    f"→ Sampled correction '{corrected_text}' (ID: {corrected_token}) from adjusted distribution"
                )
            elif accepted_count == 0:
                # All tokens rejected, no correction was made yet
                rejected_text = self.tokenizer.decode(draft_result.token_ids)
                logger.info(
                    f"✗ Rejected all {len(draft_result.token_ids)} draft tokens: '{rejected_text}'"
                )

            generated_tokens.extend(accepted_tokens)

            if accepted_count == 0 and not has_correction:
                logger.info("→ Generating fallback token from verifier model...")
                verifier_count, fallback_tokens = self._generate_from_verifier(
                    generated_tokens, count=1, temperature=generation_config.temperature
                )
                fallback_text = self.tokenizer.decode(fallback_tokens)
                fallback_ids_str = ", ".join(str(t) for t in fallback_tokens)
                logger.info(
                    f"→ Verifier sampled fallback token: '{fallback_text}' (ID: {fallback_ids_str})"
                )
                num_verifier_generated += verifier_count
                generated_tokens.extend(fallback_tokens)

        final_text = self.tokenizer.decode(generated_tokens)
        total_draft_proposed = num_draft_accepted + num_draft_rejected
        acceptance_rate = (
            num_draft_accepted / total_draft_proposed
            if total_draft_proposed > 0
            else 0.0
        )

        logger.info(
            f"Generation complete: {len(generated_tokens) - len(input_ids[0])} tokens generated"
        )
        logger.info(
            f"Draft acceptance rate: {acceptance_rate:.2%} ({num_draft_accepted}/{total_draft_proposed})"
        )
        logger.info(
            f"Breakdown: {num_draft_accepted} draft accepted, {num_draft_rejected} draft rejected, {num_verifier_generated} verifier generated"
        )

        return GenerationResult(
            text=final_text,
            token_ids=generated_tokens,
            acceptance_rate=acceptance_rate,
            num_accepted=num_draft_accepted,
            num_rejected=num_draft_rejected,
            total_tokens=len(generated_tokens) - len(input_ids[0]),
        )

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
    ) -> tuple[int, int, list[int], bool]:
        full_sequence = context_tokens + draft_tokens
        input_ids = torch.tensor([full_sequence], device=self.device)

        with torch.no_grad():
            outputs = self.model(input_ids)
            all_logits = outputs.logits[0, :, :]

        accepted_tokens = []
        num_actually_accepted = 0
        has_correction = False
        context_len = len(context_tokens)

        for i, draft_token in enumerate(draft_tokens):
            position = context_len + i - 1
            logits = all_logits[position, :] / temperature
            target_probs = torch.softmax(logits, dim=-1)

            target_prob = target_probs[draft_token].item()
            draft_prob = draft_probs[i]

            acceptance_prob = min(1.0, target_prob / draft_prob)

            if random.random() < acceptance_prob:
                accepted_tokens.append(draft_token)
                num_actually_accepted += 1
            else:
                succ = successors[i]
                q_weights = successor_weights[i]

                q = torch.zeros_like(target_probs)
                q[succ] = torch.tensor(q_weights, device=target_probs.device)

                residual = torch.clamp(target_probs - q, min=0.0)

                if residual.sum() == 0:
                    p_cond = target_probs.clone()
                    p_cond[draft_token] = 0.0
                    residual = p_cond

                if residual.sum() > 0:
                    residual = residual / residual.sum()
                    corrected_token = torch.multinomial(residual, num_samples=1).item()
                    accepted_tokens.append(corrected_token)
                    has_correction = True

                break

        rejected_count = len(draft_tokens) - num_actually_accepted
        return num_actually_accepted, rejected_count, accepted_tokens, has_correction

    def _generate_from_verifier(
        self, context_tokens: list[int], count: int, temperature: float
    ) -> tuple[int, list[int]]:
        input_ids = torch.tensor([context_tokens], device=self.device)
        generated = []

        for _ in range(count):
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits[0, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)

                next_token = torch.multinomial(probs, num_samples=1).item()
                generated.append(next_token)

                input_ids = torch.cat(
                    [input_ids, torch.tensor([[next_token]], device=self.device)], dim=1
                )

        return len(generated), generated

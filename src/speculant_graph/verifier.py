import torch
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from loguru import logger

from speculant_graph.draft_generator import DraftGenerator
from speculant_graph.config import VerifierConfig, GenerationConfig, DraftConfig


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
        draft_config: DraftConfig
    ):
        self.verifier_config = verifier_config
        self.draft_config = draft_config

        logger.info(f"Loading verifier model: {verifier_config.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            verifier_config.model_name,
            token=verifier_config.hf_token
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            verifier_config.model_name,
            token=verifier_config.hf_token
        )

        self.device = verifier_config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(f"Model loaded on device: {self.device}")

        logger.info(f"Loading knowledge graph from: {graph_path}")
        self.draft_generator = DraftGenerator.from_file(
            graph_path,
            verifier_config.model_name,
            verifier_config.hf_token
        )
        logger.info("Knowledge graph loaded successfully")

    def generate(self, prompt: str, generation_config: GenerationConfig) -> GenerationResult:
        prompt_preview = prompt if len(prompt) <= 100 else f"{prompt[:100]}..."
        logger.info(f"Generating with prompt: '{prompt_preview}'")
        logger.info(f"Config: max_tokens={generation_config.max_tokens}, "
                   f"temperature={generation_config.temperature}")

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        generated_tokens = input_ids[0].tolist()

        num_accepted = 0
        num_rejected = 0

        while len(generated_tokens) - len(input_ids[0]) < generation_config.max_tokens:
            current_text = self.tokenizer.decode(generated_tokens)

            tokens_remaining = generation_config.max_tokens - (len(generated_tokens) - len(input_ids[0]))
            draft_k = min(self.draft_config.k, tokens_remaining)

            draft_result = self.draft_generator.generate(
                prompt=current_text,
                k=draft_k,
                strategy=self.draft_config.strategy
            )

            if len(draft_result.token_ids) == 0:
                logger.debug("Draft generator returned empty sequence, generating from verifier")
                accepted_count, new_tokens = self._generate_from_verifier(
                    generated_tokens,
                    count=1,
                    temperature=generation_config.temperature
                )
                num_accepted += accepted_count
                generated_tokens.extend(new_tokens)
                continue

            accepted_count, rejected_count, accepted_tokens = self._verify_draft(
                generated_tokens,
                draft_result.token_ids,
                temperature=generation_config.temperature
            )

            num_accepted += accepted_count
            num_rejected += rejected_count

            if accepted_tokens:
                accepted_text = self.tokenizer.decode(accepted_tokens)
                logger.info(f"Verifier accepted {accepted_count}/{len(draft_result.token_ids)} tokens: '{accepted_text}'")
            else:
                logger.info(f"Verifier rejected all {len(draft_result.token_ids)} draft tokens")

            generated_tokens.extend(accepted_tokens)

            if accepted_count == 0:
                logger.debug("No tokens accepted, generating from verifier as fallback")
                fallback_count, fallback_tokens = self._generate_from_verifier(
                    generated_tokens,
                    count=1,
                    temperature=generation_config.temperature
                )
                fallback_text = self.tokenizer.decode(fallback_tokens)
                logger.info(f"Verifier generated fallback token: '{fallback_text}'")
                num_accepted += fallback_count
                generated_tokens.extend(fallback_tokens)

        final_text = self.tokenizer.decode(generated_tokens)
        total_proposed = num_accepted + num_rejected
        acceptance_rate = num_accepted / total_proposed if total_proposed > 0 else 0.0

        logger.info(f"Generation complete: {len(generated_tokens) - len(input_ids[0])} tokens generated")
        logger.info(f"Acceptance rate: {acceptance_rate:.2%} ({num_accepted}/{total_proposed})")

        return GenerationResult(
            text=final_text,
            token_ids=generated_tokens,
            acceptance_rate=acceptance_rate,
            num_accepted=num_accepted,
            num_rejected=num_rejected,
            total_tokens=len(generated_tokens) - len(input_ids[0])
        )

    def _verify_draft(
        self,
        context_tokens: list[int],
        draft_tokens: list[int],
        temperature: float
    ) -> tuple[int, int, list[int]]:
        input_ids = torch.tensor([context_tokens], device=self.device)

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[0, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)

        accepted_tokens = []
        rejected_count = 0

        for draft_token in draft_tokens:
            draft_prob = probs[draft_token].item()

            if draft_prob > self.verifier_config.acceptance_threshold:
                accepted_tokens.append(draft_token)

                if len(accepted_tokens) < len(draft_tokens):
                    context_tokens.append(draft_token)
                    input_ids = torch.tensor([context_tokens], device=self.device)

                    with torch.no_grad():
                        outputs = self.model(input_ids)
                        logits = outputs.logits[0, -1, :] / temperature
                        probs = torch.softmax(logits, dim=-1)
            else:
                rejected_count += 1
                break

        return len(accepted_tokens), rejected_count, accepted_tokens

    def _generate_from_verifier(
        self,
        context_tokens: list[int],
        count: int,
        temperature: float
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

                input_ids = torch.cat([
                    input_ids,
                    torch.tensor([[next_token]], device=self.device)
                ], dim=1)

        return len(generated), generated

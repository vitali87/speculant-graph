import argparse
import random
import sys
import time
from typing import Iterator

import torch
from loguru import logger
from pydantic import BaseModel, computed_field
from transformers import logging as transformers_logging

modules_to_remove = [
    k for k in list(sys.modules.keys()) if k.startswith("speculant_graph")
]
for mod in modules_to_remove:
    del sys.modules[mod]

from speculant_graph import SpeculativeDecoder  # noqa: E402
from speculant_graph.config import DraftConfig, GenerationConfig, VerifierConfig  # noqa: E402

transformers_logging.set_verbosity_error()


class BenchmarkResult(BaseModel):
    text: str
    duration: float
    num_tokens: int

    @computed_field
    @property
    def tokens_per_second(self) -> float:
        return self.num_tokens / self.duration if self.duration > 0 else 0.0


class SpeculativeResult(BaseModel):
    text: str
    duration: float
    num_tokens: int
    acceptance_rate: float
    num_accepted: int
    num_rejected: int
    position_acceptance_counts: dict[int, int] = {}
    position_proposal_counts: dict[int, int] = {}

    @computed_field
    @property
    def tokens_per_second(self) -> float:
        return self.num_tokens / self.duration if self.duration > 0 else 0.0


class BenchmarkComparison(BaseModel):
    prompt: str
    max_tokens: int
    native: BenchmarkResult
    speculative: SpeculativeResult

    @computed_field
    @property
    def speedup(self) -> float:
        if self.native.tokens_per_second == 0:
            return 0.0
        return self.speculative.tokens_per_second / self.native.tokens_per_second

    def display(self) -> None:
        logger.info("\n" + "=" * 70)
        logger.info("BENCHMARK RESULTS")
        logger.info("=" * 70)
        logger.info(f"\nPrompt: {self.prompt!r}")
        logger.info(f"Tokens requested: {self.max_tokens}")

        logger.info("\n📊 NATIVE DECODING")
        logger.info(f"  ⏱️  Time: {self.native.duration:.2f}s")
        logger.info(f"  ⚡ Speed: {self.native.tokens_per_second:.2f} tokens/sec")
        logger.info(f"  📝 Output: {self.native.text}")

        logger.info("\n🚀 SPECULATIVE DECODING")
        logger.info(f"  ⏱️  Time: {self.speculative.duration:.2f}s")
        logger.info(f"  ⚡ Speed: {self.speculative.tokens_per_second:.2f} tokens/sec")
        logger.info(
            f"  ✅ Acceptance: {self.speculative.acceptance_rate:.2%} "
            f"({self.speculative.num_accepted}/{self.speculative.num_accepted + self.speculative.num_rejected})"
        )

        if self.speculative.position_proposal_counts:
            logger.info("\n  📍 Position-wise acceptance:")
            max_pos = max(self.speculative.position_proposal_counts.keys())
            for pos in range(max_pos + 1):
                proposals = self.speculative.position_proposal_counts.get(pos, 0)
                acceptances = self.speculative.position_acceptance_counts.get(pos, 0)
                if proposals > 0:
                    rate = acceptances / proposals
                    logger.info(
                        f"     Position {pos}: {acceptances}/{proposals} ({rate:.2%})"
                    )

        logger.info(f"  📝 Output: {self.speculative.text}")

        logger.info("\n" + "=" * 70)
        logger.info(f"🎯 SPEEDUP: {self.speedup:.2f}x faster")
        logger.info("=" * 70 + "\n")


def _set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _prepare_input_ids(decoder: SpeculativeDecoder, prompt: str) -> torch.Tensor:
    tokenizer = decoder.tokenizer
    input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")[
        "input_ids"
    ].to(decoder.device)

    if len(input_ids[0]) > 0:
        return input_ids

    starter_token = (
        tokenizer.bos_token_id
        or tokenizer.eos_token_id
        or tokenizer.pad_token_id
        or decoder.draft_generator.get_most_frequent_token()
    )

    return torch.tensor([[starter_token]], device=decoder.device)


def _native_decode_stream(
    decoder: SpeculativeDecoder, prompt: str, generation_config: GenerationConfig
) -> Iterator[tuple[str, list[int]]]:
    _set_seed(generation_config.seed)
    tokenizer = decoder.tokenizer
    device = decoder.device
    input_ids = _prepare_input_ids(decoder, prompt)
    attention_mask = torch.ones(
        (1, input_ids.shape[1]), device=device, dtype=torch.long
    )

    with torch.no_grad():
        outputs = decoder.model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=True
        )

    past_key_values = outputs.past_key_values
    logits = outputs.logits[:, -1, :]
    generated_tokens: list[int] = []

    for _ in range(generation_config.max_tokens):
        probs = torch.softmax(logits.squeeze(0) / generation_config.temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        generated_tokens.append(next_token)
        text = tokenizer.decode([next_token], skip_special_tokens=True)
        yield text, generated_tokens.copy()

        token_tensor = torch.tensor([[next_token]], device=device)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), device=device, dtype=torch.long)],
            dim=1,
        )

        with torch.no_grad():
            outputs = decoder.model(
                input_ids=token_tensor,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )

        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]


def _run_native_benchmark(
    decoder: SpeculativeDecoder, prompt: str, generation_config: GenerationConfig
) -> BenchmarkResult:
    logger.info("Running native decoding...")
    _sync_cuda()
    start = time.perf_counter()

    print("Native: ", end="", flush=True)
    native_tokens: list[int] = []
    text_parts: list[str] = []

    for text_part, tokens in _native_decode_stream(decoder, prompt, generation_config):
        print(text_part, end="", flush=True)
        native_tokens = tokens
        text_parts.append(text_part)

    print("\n", flush=True)
    _sync_cuda()
    duration = time.perf_counter() - start

    return BenchmarkResult(
        text="".join(text_parts), duration=duration, num_tokens=len(native_tokens)
    )


def _run_speculative_benchmark(
    decoder: SpeculativeDecoder, prompt: str, generation_config: GenerationConfig
) -> SpeculativeResult:
    logger.info("Running speculative decoding...")
    _sync_cuda()
    start = time.perf_counter()

    print("Speculative: ", end="", flush=True)
    spec_result = None

    for item in decoder.generate_stream(prompt, generation_config):
        if isinstance(item, tuple):
            text_part, _ = item
            print(text_part, end="", flush=True)
        else:
            spec_result = item

    print("\n", flush=True)
    _sync_cuda()
    duration = time.perf_counter() - start

    if spec_result is None:
        raise RuntimeError("No result returned from speculative decoding")

    return SpeculativeResult(
        text=spec_result.text,
        duration=duration,
        num_tokens=spec_result.total_tokens,
        acceptance_rate=spec_result.acceptance_rate,
        num_accepted=spec_result.num_accepted,
        num_rejected=spec_result.num_rejected,
        position_acceptance_counts=spec_result.position_acceptance_counts,
        position_proposal_counts=spec_result.position_proposal_counts,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark graph speculative decoding against native decoding.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--graph-path", required=True, help="Path to the n-gram graph")
    parser.add_argument(
        "--prompt",
        default="Explain the main idea behind graph-based speculative decoding.",
        help="Benchmark prompt",
    )
    parser.add_argument(
        "--model-name",
        default="ByteDance-Seed/Seed-OSS-36B-Instruct",
        help="Verifier model from HuggingFace",
    )
    parser.add_argument("--hf-token", help="HuggingFace token for private models")
    parser.add_argument(
        "--device", help="Torch device (cuda/cpu), auto-detected if not specified"
    )
    parser.add_argument("--device-map", default="auto", help="Model device map")
    parser.add_argument(
        "--torch-dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Model precision",
    )
    parser.add_argument("--max-tokens", type=int, default=10, help="Tokens to generate")
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Sampling temperature"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--draft-k", type=int, default=8, help="Tokens per speculative draft"
    )
    parser.add_argument(
        "--draft-strategy",
        choices=["greedy", "sampling"],
        default="greedy",
        help="Draft strategy",
    )
    parser.add_argument("--verbose", action="store_true", help="Detailed logging")
    return parser.parse_args()


def _setup_decoder(args: argparse.Namespace) -> SpeculativeDecoder:
    device_map = None if args.device_map.lower() == "none" else args.device_map
    torch_dtype = None if args.torch_dtype.lower() == "none" else args.torch_dtype

    verifier_config = VerifierConfig(
        model_name=args.model_name,
        device=args.device,
        hf_token=args.hf_token,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    draft_config = DraftConfig(k=args.draft_k, strategy=args.draft_strategy)

    return SpeculativeDecoder(
        graph_path=args.graph_path,
        verifier_config=verifier_config,
        draft_config=draft_config,
    )


def _configure_logging(verbose: bool) -> None:
    if not verbose:
        logger.remove()
        logger.add(
            sys.stderr,
            filter=lambda record: record["name"] == "__main__",
            format="{message}",
            level="INFO",
        )


def main() -> None:
    args = _parse_args()
    _configure_logging(args.verbose)

    logger.info("Loading speculative decoder...")
    decoder = _setup_decoder(args)

    generation_config = GenerationConfig(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=args.seed,
    )

    native_result = _run_native_benchmark(decoder, args.prompt, generation_config)
    speculative_result = _run_speculative_benchmark(
        decoder, args.prompt, generation_config
    )

    comparison = BenchmarkComparison(
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        native=native_result,
        speculative=speculative_result,
    )
    comparison.display()


if __name__ == "__main__":
    main()

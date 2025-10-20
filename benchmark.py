#!/usr/bin/env python3

import argparse
import random
import sys
import time

import torch
from loguru import logger

from speculant_graph import SpeculativeDecoder
from speculant_graph.config import DraftConfig, GenerationConfig, VerifierConfig


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

    starter_token = None
    if tokenizer.bos_token_id is not None:
        starter_token = tokenizer.bos_token_id
    elif tokenizer.eos_token_id is not None:
        starter_token = tokenizer.eos_token_id
    elif tokenizer.pad_token_id is not None:
        starter_token = tokenizer.pad_token_id
    else:
        starter_token = decoder.draft_generator.get_most_frequent_token()

    return torch.tensor([[starter_token]], device=decoder.device)


def _native_decode(
    decoder: SpeculativeDecoder, prompt: str, generation_config: GenerationConfig
) -> tuple[str, list[int]]:
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

        token_tensor = torch.tensor([[next_token]], device=device)
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones((1, 1), device=device, dtype=torch.long),
            ],
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

    text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return text, generated_tokens


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark graph speculative decoding against native decoding."
    )
    parser.add_argument(
        "--graph-path",
        required=True,
        help="Path to the n-gram graph built by the draft generator.",
    )
    parser.add_argument(
        "--prompt",
        default="Explain the main idea behind graph-based speculative decoding.",
        help="Prompt used for the benchmark run.",
    )
    parser.add_argument(
        "--model-name",
        default="meta-llama/Llama-3.2-3B",
        help="Verifier model name to load from Hugging Face.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional Hugging Face token for private models.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device (e.g. 'cuda', 'cuda:0', 'cpu'). Overrides auto detection.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Device map for model loading (passed to from_pretrained).",
    )
    parser.add_argument(
        "--torch-dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype for the verifier model.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=10,
        help="Number of new tokens to generate for each run.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--draft-k",
        type=int,
        default=8,
        help="Number of tokens proposed per speculative draft step.",
    )
    parser.add_argument(
        "--draft-strategy",
        choices=["greedy", "sampling"],
        default="greedy",
        help="Draft generation strategy.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed logging from the speculative decoder stack.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="ERROR")

    device_map = args.device_map
    if device_map is not None and device_map.lower() == "none":
        device_map = None

    torch_dtype = args.torch_dtype
    if torch_dtype is not None and torch_dtype.lower() == "none":
        torch_dtype = None

    verifier_config = VerifierConfig(
        model_name=args.model_name,
        device=args.device,
        hf_token=args.hf_token,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    draft_config = DraftConfig(k=args.draft_k, strategy=args.draft_strategy)
    generation_config = GenerationConfig(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=args.seed,
    )

    print("Loading speculative decoder...")
    decoder = SpeculativeDecoder(
        graph_path=args.graph_path,
        verifier_config=verifier_config,
        draft_config=draft_config,
    )

    # Native decoding
    print("\nRunning native verifier decoding...")
    _sync_cuda()
    native_start = time.perf_counter()
    native_text, native_tokens = _native_decode(decoder, args.prompt, generation_config)
    _sync_cuda()
    native_duration = time.perf_counter() - native_start

    tokens_per_second_native = (
        len(native_tokens) / native_duration if native_duration > 0 else 0.0
    )

    # Speculative decoding
    print("Running graph speculative decoding...")
    _sync_cuda()
    spec_start = time.perf_counter()
    spec_result = decoder.generate(args.prompt, generation_config)
    _sync_cuda()
    spec_duration = time.perf_counter() - spec_start

    tokens_per_second_spec = (
        spec_result.total_tokens / spec_duration if spec_duration > 0 else 0.0
    )

    print("\n=== Benchmark Results ===")
    print(f"Prompt: {args.prompt!r}")
    print(f"Tokens requested: {args.max_tokens}")
    print("\nNative decoding:")
    print(f"  Time: {native_duration:.2f}s")
    print(f"  Tokens/sec: {tokens_per_second_native:.2f}")
    print(f"  Output: {native_text}")

    print("\nSpeculative decoding:")
    print(f"  Time: {spec_duration:.2f}s")
    print(f"  Tokens/sec: {tokens_per_second_spec:.2f}")
    print(
        f"  Draft acceptance rate: {spec_result.acceptance_rate:.2%} "
        f"({spec_result.num_accepted} accepted / "
        f"{spec_result.num_accepted + spec_result.num_rejected} proposed)"
    )
    print(f"  Output: {spec_result.text}")


if __name__ == "__main__":
    main()

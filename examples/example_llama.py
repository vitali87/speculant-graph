import os
import argparse
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from loguru import logger

from speculant_graph import (
    GraphBuilder,
    SpeculativeDecoder,
    GraphConfig,
    DraftConfig,
    VerifierConfig,
    GenerationConfig,
)


def main():
    parser = argparse.ArgumentParser(description="Speculative Graph Decoding Example with Llama-3.2-3B")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum number of tokens to generate (default: 50)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Sampling temperature (default: 0.9)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="+",
        default=["A termination clause outlines the circumstances under which"],
        help="Prompt(s) to generate from (default: sample legal prompt)"
    )
    args = parser.parse_args()

    logger.info("Starting Speculative Graph Decoding Example with Llama-3.2-3B")

    MODEL_NAME = "meta-llama/Llama-3.2-3B"
    HF_TOKEN = os.getenv("HF_TOKEN")

    corpus_dir = Path(__file__).parent / "corpus"
    corpus_files = list(corpus_dir.glob("*.txt"))
    graph_path = Path(__file__).parent / "llama_knowledge_graph.pkl"

    if not graph_path.exists():
        logger.info(
            f"Building knowledge graph from corpus using {MODEL_NAME} tokenizer..."
        )
        graph_config = GraphConfig(tokenizer_name=MODEL_NAME, hf_token=HF_TOKEN)
        builder = GraphBuilder(
            tokenizer_name=graph_config.tokenizer_name,
            max_order=graph_config.max_order,
            chunk_size=graph_config.chunk_size,
            hf_token=graph_config.hf_token,
        )

        builder.build_from_files([str(f) for f in corpus_files])
        builder.save(str(graph_path))
    else:
        logger.info(f"Using existing graph: {graph_path}")

    verifier_config = VerifierConfig(
        model_name=MODEL_NAME, hf_token=HF_TOKEN
    )
    draft_config = DraftConfig(k=8, strategy="greedy")

    decoder = SpeculativeDecoder(
        graph_path=str(graph_path),
        verifier_config=verifier_config,
        draft_config=draft_config,
    )

    # Join all prompt arguments into a single string
    prompts = [" ".join(args.prompt)]

    for prompt in prompts:
        logger.info(f"\nPrompt: {prompt}")

        generation_config = GenerationConfig(max_tokens=args.max_tokens, temperature=args.temperature)
        result = decoder.generate(prompt, generation_config)

        logger.info(f"Generated text: {result.text}")
        logger.info(f"Acceptance rate: {result.acceptance_rate:.2%}")
        logger.info(
            f"Tokens: accepted={result.num_accepted}, rejected={result.num_rejected}"
        )


if __name__ == "__main__":
    main()

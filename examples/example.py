import os
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
    logger.info("Starting Speculative Graph Decoding Example")

    corpus_dir = Path(__file__).parent / "corpus"
    corpus_files = list(corpus_dir.glob("*.txt"))
    graph_path = Path(__file__).parent / "knowledge_graph.pkl"

    if not graph_path.exists():
        logger.info("Building knowledge graph from corpus...")
        graph_config = GraphConfig()
        builder = GraphBuilder(
            tokenizer_name=graph_config.tokenizer_name,
            chunk_size=graph_config.chunk_size,
        )

        graph = builder.build_from_files([str(f) for f in corpus_files])
        builder.save(str(graph_path))
    else:
        logger.info(f"Using existing graph: {graph_path}")

    verifier_config = VerifierConfig(acceptance_threshold=0.6)
    draft_config = DraftConfig(k=8, strategy="greedy")

    decoder = SpeculativeDecoder(
        graph_path=str(graph_path),
        verifier_config=verifier_config,
        draft_config=draft_config,
    )

    prompts = [
        "What is a force majeure clause?",
        "Explain indemnification clauses.",
        "What are liquidated damages?",
    ]

    for prompt in prompts:
        logger.info(f"\nPrompt: {prompt}")

        generation_config = GenerationConfig(max_tokens=50, temperature=0.9)
        result = decoder.generate(prompt, generation_config)

        logger.info(f"Generated text: {result.text}")
        logger.info(f"Acceptance rate: {result.acceptance_rate:.2%}")
        logger.info(
            f"Tokens: accepted={result.num_accepted}, rejected={result.num_rejected}"
        )


if __name__ == "__main__":
    main()

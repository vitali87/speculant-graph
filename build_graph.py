import argparse
import os
import sys
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from loguru import logger

from speculant_graph import GraphBuilder


def main():
    parser = argparse.ArgumentParser(
        description="Build multi-order n-gram graph from corpus files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build_graph.py --corpus-dir examples/corpus --output graph.pkl

  python build_graph.py --corpus-dir examples/corpus --output llama_graph.pkl \\
    --model-name meta-llama/Llama-3.2-3B

  python build_graph.py --corpus-dir examples/corpus --output graph.pkl \\
    --max-order 8

  python build_graph.py --corpus-files corpus1.txt corpus2.txt \\
    --output graph.pkl

  python build_graph.py --corpus-dir examples/corpus --output graph.pkl \\
    --model-name meta-llama/Llama-3.1-8B --hf-token $HF_TOKEN
        """,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--corpus-dir",
        type=Path,
        help="Directory containing corpus .txt files",
    )
    input_group.add_argument(
        "--corpus-files",
        nargs="+",
        type=Path,
        help="List of corpus files to process",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output path for the graph file (.pkl)",
    )

    parser.add_argument(
        "--max-order",
        type=int,
        default=5,
        help="Maximum n-gram order (1-10, default: 5)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10000,
        help="Processing chunk size (default: 10000)",
    )

    parser.add_argument(
        "--model-name",
        default="meta-llama/Llama-3.2-3B",
        help="HuggingFace model/tokenizer name (must match verifier model!)",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace token for gated models",
    )
    parser.add_argument(
        "--download-mode",
        choices=["auto", "hf_transfer", "default"],
        default="auto",
        help="Download acceleration mode",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress all logging except errors",
    )

    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force rebuild even if output file exists",
    )

    args = parser.parse_args()

    if args.quiet:
        logger.remove()
        logger.add(sys.stderr, level="ERROR")
    elif not args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="INFO")

    if args.output.exists() and not args.force:
        logger.error(f"Output file already exists: {args.output}")
        logger.error("Use --force to overwrite")
        sys.exit(1)

    if args.corpus_dir:
        if not args.corpus_dir.exists():
            logger.error(f"Corpus directory not found: {args.corpus_dir}")
            sys.exit(1)

        corpus_files = list(args.corpus_dir.glob("*.txt"))
        if not corpus_files:
            logger.error(f"No .txt files found in: {args.corpus_dir}")
            sys.exit(1)

        logger.info(f"Found {len(corpus_files)} corpus files in {args.corpus_dir}")
    else:
        corpus_files = args.corpus_files
        for f in corpus_files:
            if not f.exists():
                logger.error(f"Corpus file not found: {f}")
                sys.exit(1)

        logger.info(f"Processing {len(corpus_files)} corpus files")

    if not (1 <= args.max_order <= 10):
        logger.error("max_order must be between 1 and 10")
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Building multi-order n-gram graph")
    logger.info("=" * 60)
    logger.info(f"Model/Tokenizer: {args.model_name}")
    logger.info(f"Max order: {args.max_order}")
    logger.info(f"Chunk size: {args.chunk_size}")
    logger.info(f"Corpus files: {len(corpus_files)}")
    logger.info(f"Output: {args.output}")
    logger.info("=" * 60)

    try:
        builder = GraphBuilder(
            tokenizer_name=args.model_name,
            max_order=args.max_order,
            chunk_size=args.chunk_size,
            hf_token=args.hf_token,
            download_mode=args.download_mode,
        )

        total_size = 0
        for f in corpus_files:
            size = f.stat().st_size
            total_size += size
            logger.debug(f"  {f.name}: {size / 1024 / 1024:.2f} MB")

        logger.info(f"Total corpus size: {total_size / 1024 / 1024:.2f} MB")

        graph = builder.build_from_files([str(f) for f in corpus_files])

        builder.save(str(args.output))

        logger.info("=" * 60)
        logger.info("✅ Graph built successfully!")
        logger.info("=" * 60)
        logger.info(f"Output: {args.output}")
        logger.info(f"File size: {args.output.stat().st_size / 1024 / 1024:.2f} MB")
        logger.info(f"Nodes: {graph.number_of_nodes():,}")
        logger.info(f"Edges: {graph.number_of_edges():,}")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Run benchmark:")
        logger.info(f"     python benchmark.py --graph-path {args.output}")
        logger.info("  2. Use in your code:")
        logger.info("     from speculant_graph import SpeculativeDecoder")
        logger.info(
            f"     decoder = SpeculativeDecoder(graph_path='{args.output}', ...)"
        )

    except Exception as e:
        logger.error(f"Failed to build graph: {e}")
        import traceback

        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

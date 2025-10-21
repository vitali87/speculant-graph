import argparse
from pathlib import Path

from loguru import logger


def download_wikipedia(output_dir: Path, max_docs: int = 10000):
    from datasets import load_dataset

    logger.info(f"Downloading Wikipedia articles (max {max_docs})...")
    logger.info("Note: This may take a few minutes on first run...")

    ds = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,
        trust_remote_code=False,
    )

    output_file = output_dir / "wikipedia.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for i, example in enumerate(ds):
            if i >= max_docs:
                break
            f.write(example["text"] + "\n\n")
            if (i + 1) % 1000 == 0:
                logger.info(f"Processed {i + 1} articles...")

    logger.success(f"Saved {max_docs} Wikipedia articles to {output_file}")


def download_openwebtext(output_dir: Path, max_docs: int = 10000):
    from datasets import load_dataset

    logger.info(f"Downloading OpenWebText (max {max_docs})...")
    ds = load_dataset("openwebtext", split="train", streaming=True)

    output_file = output_dir / "openwebtext.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for i, example in enumerate(ds):
            if i >= max_docs:
                break
            f.write(example["text"] + "\n\n")
            if (i + 1) % 1000 == 0:
                logger.info(f"Processed {i + 1} documents...")

    logger.success(f"Saved {max_docs} OpenWebText documents to {output_file}")


def download_c4(output_dir: Path, max_docs: int = 10000):
    from datasets import load_dataset

    logger.info(f"Downloading C4 corpus (max {max_docs})...")
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)

    output_file = output_dir / "c4.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for i, example in enumerate(ds):
            if i >= max_docs:
                break
            f.write(example["text"] + "\n\n")
            if (i + 1) % 1000 == 0:
                logger.info(f"Processed {i + 1} documents...")

    logger.success(f"Saved {max_docs} C4 documents to {output_file}")


def download_bookcorpus(output_dir: Path, max_docs: int = 5000):
    from datasets import load_dataset

    logger.info(f"Downloading BookCorpus (max {max_docs})...")
    ds = load_dataset("bookcorpus", split="train", streaming=True)

    output_file = output_dir / "bookcorpus.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for i, example in enumerate(ds):
            if i >= max_docs:
                break
            f.write(example["text"] + "\n\n")
            if (i + 1) % 1000 == 0:
                logger.info(f"Processed {i + 1} documents...")

    logger.success(f"Saved {max_docs} BookCorpus documents to {output_file}")


def run_downloads(corpus: str, output_dir: Path, max_docs: int):
    if corpus in ("wikipedia", "all"):
        download_wikipedia(output_dir, max_docs)
    if corpus in ("openwebtext", "all"):
        download_openwebtext(output_dir, max_docs)
    if corpus in ("c4", "all"):
        download_c4(output_dir, max_docs)
    if corpus in ("bookcorpus", "all"):
        download_bookcorpus(output_dir, max_docs // 2)


def main():
    parser = argparse.ArgumentParser(
        description="Download corpus data for n-gram graph building"
    )
    parser.add_argument(
        "--corpus",
        choices=["wikipedia", "openwebtext", "c4", "bookcorpus", "all"],
        default="wikipedia",
        help="Which corpus to download",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/corpus"),
        help="Output directory for corpus files",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=10000,
        help="Maximum number of documents to download",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading to: {args.output_dir}")
    logger.info(f"Max documents: {args.max_docs}\n")

    try:
        run_downloads(args.corpus, args.output_dir, args.max_docs)
    except Exception as e:
        logger.error(f"Error downloading corpus: {e}")
        return

    logger.success("Download complete!")
    logger.info("\nNext steps:")
    logger.info("1. Build graph: python examples/example.py")
    logger.info("2. Or use GraphBuilder directly:")
    logger.info("   from speculant_graph import GraphBuilder")
    path = args.output_dir
    logger.info(f"   builder.build_from_files(list({path}.glob('*.txt')))")


if __name__ == "__main__":
    main()

# Getting Started

## Installation

```bash
uv sync
```

### Development Setup

```bash
uv sync --extra dev
pre-commit install
pre-commit install --hook-type commit-msg
```

## Step 1: Download Corpus

```bash
python download_corpus.py --corpus wikipedia --max-docs 20000
```

## Step 2: Build Graph

```bash
python build_graph.py \
  --corpus-dir examples/corpus \
  --output graph.pkl \
  --model-name ByteDance-Seed/Seed-OSS-36B-Instruct \
  --max-order 5
```

## Step 3: Generate

```python
from speculant_graph import (
    SpeculativeDecoder, DraftConfig, VerifierConfig, GenerationConfig
)

decoder = SpeculativeDecoder(
    graph_path="graph.pkl",
    verifier_config=VerifierConfig(model_name="ByteDance-Seed/Seed-OSS-36B-Instruct"),
    draft_config=DraftConfig(k=8, strategy="greedy"),
)

result = decoder.generate(
    prompt="What is contract law?",
    generation_config=GenerationConfig(max_tokens=100, temperature=0.8),
)
print(result.text)
print(f"Acceptance rate: {result.acceptance_rate:.2%}")
```

## Step 4: Benchmark

```bash
python benchmark.py \
  --graph-path graph.pkl \
  --model-name ByteDance-Seed/Seed-OSS-36B-Instruct \
  --max-tokens 100
```

## Server Mode

Avoid reloading model weights between requests:

```bash
uv sync --extra server
uv run --extra server server/app.py --graph-path graph.pkl
```

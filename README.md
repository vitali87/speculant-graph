<p align="center">
  <img src="assets/speculant-graph-logo-mark.svg#gh-light-mode-only" alt="Speculant Graph Logo" width="300" />
  <img src="assets/speculant-graph-logo-mark-dark.svg#gh-dark-mode-only" alt="Speculant Graph Logo" width="300" />
</p>

<h1 align="center">Speculant Graph</h1>
<h3 align="center">Graph drafts, LLM verifies</h3>

<p align="center">
  <a href="https://github.com/vitali87/speculant-graph/stargazers"><img src="https://img.shields.io/github/stars/vitali87/speculant-graph?style=social" alt="GitHub stars"></a>
  <a href="https://github.com/vitali87/speculant-graph/network/members"><img src="https://img.shields.io/github/forks/vitali87/speculant-graph?style=social" alt="GitHub forks"></a>
  <a href="https://github.com/vitali87/speculant-graph/watchers"><img src="https://img.shields.io/github/watchers/vitali87/speculant-graph?style=social" alt="GitHub watchers"></a>
  <a href="https://github.com/vitali87/speculant-graph/issues"><img src="https://img.shields.io/github/issues/vitali87/speculant-graph" alt="GitHub issues"></a>
  <a href="https://github.com/vitali87/speculant-graph/blob/main/LICENSE"><img src="https://img.shields.io/github/license/vitali87/speculant-graph" alt="License"></a>
  <img src="https://img.shields.io/github/languages/top/vitali87/speculant-graph" alt="Top language">
  <img src="https://img.shields.io/github/last-commit/vitali87/speculant-graph" alt="Last commit">
</p>

https://github.com/vitali87/speculant-graph/assets/ai_spec_2_final.mov

# Speculative Graph Decoding

Novel approach to speculative decoding using multi-order n-gram graphs as draft models instead of small LLMs.

## Overview

Traditional speculative decoding uses a small draft model to propose tokens that a large verifier model accepts or rejects. This project replaces the draft model with a **multi-order n-gram graph** built from domain-specific text corpora.

### Key Innovation

- **Multi-order Markov Chains**: Adaptively uses 1st through 5th order 
context for accurate predictions
- **Attentive Context Mixing** (default): Attention mechanism that blends 
multiple n-gram orders for smoother, more robust proposals
- **Zero training**: No need to train or maintain a separate draft model
- **Domain-specific**: Graph captures patterns from user-supplied corpora 
(law, finance, healthcare, etc.)
- **Transparent**: All transitions are traceable to source text with O(1) 
context lookup

## Installation

```bash
uv sync
```

### Development Setup

This project uses pre-commit hooks to maintain code quality:
- **uv-lock**: Ensures the lockfile is up-to-date
- **Ruff**: Linting and code formatting
- **Conventional Commits**: Validates commit message format

Setup pre-commit hooks:

```bash
uv sync --extra dev
pre-commit install
pre-commit install --hook-type commit-msg
```

Run hooks manually on all files:

```bash
pre-commit run --all-files
```

## Quick Start

### 1. Download Corpus

For best results, download a substantial corpus. With **20,000 Wikipedia articles**, you can achieve **2-5x speedup**:

```bash
# Download 20k Wikipedia articles (~128MB corpus)
python download_corpus.py --corpus wikipedia --max-docs 20000
```

### 2. Build Multi-Order N-gram Graph

```bash
# Build graph for large model (recommended: ByteDance-Seed/Seed-OSS-36B-Instruct)
python build_graph.py \
  --corpus-dir examples/corpus \
  --output graph.pkl \
  --model-name ByteDance-Seed/Seed-OSS-36B-Instruct \
  --max-order 5
```

Or programmatically:

```python
from speculant_graph import GraphBuilder

builder = GraphBuilder(
    tokenizer_name="ByteDance-Seed/Seed-OSS-36B-Instruct",
    max_order=5,
    chunk_size=10000
)

graph = builder.build_from_files(["examples/corpus/wikipedia.txt"])
builder.save("graph.pkl")
```

### 3. Generate with Speculative Decoding

```python
from speculant_graph import (
    SpeculativeDecoder,
    DraftConfig,
    VerifierConfig,
    GenerationConfig
)

decoder = SpeculativeDecoder(
    graph_path="graph.pkl",
    verifier_config=VerifierConfig(
        model_name="ByteDance-Seed/Seed-OSS-36B-Instruct"
    ),
    draft_config=DraftConfig(k=8, strategy="greedy")
)

result = decoder.generate(
    prompt="What is contract law?",
    generation_config=GenerationConfig(max_tokens=100, temperature=0.8)
)

print(result.text)
print(f"Acceptance rate: {result.acceptance_rate:.2%}")
```

### 4. Benchmark Performance

Run the benchmark to see the speedup:

```bash
python benchmark.py \
  --graph-path graph.pkl \
  --model-name ByteDance-Seed/Seed-OSS-36B-Instruct \
  --max-tokens 100 \
  --prompt "What is contract law?"
```

**Example Results** (20k Wikipedia corpus):
```
Native decoding:
  Time: 41.29s
  Tokens/sec: 2.42

Speculative decoding:
  Time: 16.69s
  Tokens/sec: 5.99
  Draft acceptance rate: 21.00%
  Position 0 acceptance: 21/100 (21.00%)

============================================================
Speedup: 2.47x faster than native decoding
============================================================
```

With different prompts and settings, speedups range from **2-5x**!

### 3. Attentive Context Mixing (Default)

**By default**, the system uses **attentive context mixing** which blends
multiple order contexts with attention weights for more robust proposals.

**How it works:** Computes attention weights via softmax, then mixes all 
matched order distributions. Example weights:
- Order-5: 63.6% (most specific)
- Order-4: 23.4%
- Order-3: 8.6%
- Order-2: 3.2%
- Order-1: 1.2% (most general)

**Tune mixing behavior:**
```python
draft_config = DraftConfig(
    k=8,
    strategy="greedy",
    attentive_mix=True,       # Default: True
    order_bias=1.0,           # β: preference for higher orders
    mix_temperature=1.0,      # τ: softmax temperature
    reliability_weight=1.0,   # Weight for log-count reliability
    entropy_penalty=0.5,      # Penalty for uncertain distributions
)
```

**Configuration:**
- `order_bias` (β): Controls preference for higher orders
  - `0.5`: Gentle preference (more mixing)
  - `1.0`: Balanced (default)
  - `2.0`: Strong preference (less mixing)
- `mix_temperature` (τ): Controls sharpness of attention
  - `0.5`: Sharp (winner-take-all)
  - `1.0`: Balanced (default)
  - `2.0`: Soft (more uniform)
- `reliability_weight`: Weight for log-count reliability term
  - Higher values → favor well-supported contexts
  - Default: `1.0`
- `entropy_penalty`: Penalty coefficient for distribution entropy
  - Higher values → favor confident (peaked) distributions
  - Default: `0.5`

**Disable mixing (use single highest-order context):**
```python
draft_config = DraftConfig(
    k=8,
    strategy="greedy",
    attentive_mix=False  # Use original single-order matching
)
```

**Why attentive mixing is default:**
- ✅ More robust, less brittle drafts
- ✅ Better acceptance rates with varied corpora
- ✅ Graceful handling of sparse high-order contexts
- ✅ Only ~5-10% overhead vs significant quality gains

**When to disable:**
- Corpus is very uniform with consistent patterns
- Need absolute fastest performance
- High-order contexts already have excellent coverage

## Using Different Models

**The system works with ANY HuggingFace model!** You can use Llama, Mistral, Qwen, GPT-OSS, or any other model.

### Important: Tokenizer Alignment

⚠️ **Critical:** The tokenizer used to build the graph MUST match the verifier model's tokenizer. Otherwise, token IDs won't align and drafts will be meaningless.

**Key Rule:** Same tokenizer for both graph building AND verification. If you use `meta-llama/Llama-3.1-8B` to build the graph, you MUST use `meta-llama/Llama-3.1-8B` as the verifier model.

### Example: Using Llama 3

```python
from speculant_graph import (
    GraphBuilder,
    SpeculativeDecoder,
    GraphConfig,
    VerifierConfig,
    DraftConfig,
    GenerationConfig
)

MODEL_NAME = "meta-llama/Llama-3.1-8B"

# Build graph with Llama tokenizer
graph_config = GraphConfig(tokenizer_name=MODEL_NAME)
builder = GraphBuilder(
    tokenizer_name=graph_config.tokenizer_name,
    chunk_size=graph_config.chunk_size
)
graph = builder.build_from_files(["corpus.txt"])
builder.save("llama_graph.pkl")

# Use Llama for verification
verifier_config = VerifierConfig(
    model_name=MODEL_NAME
)
decoder = SpeculativeDecoder(
    graph_path="llama_graph.pkl",  # Must match saved filename above
    verifier_config=verifier_config,
    draft_config=DraftConfig(k=8, strategy="greedy")
)

result = decoder.generate(
    prompt="Your prompt here",
    generation_config=GenerationConfig(max_tokens=100, temperature=0.9)
)
```

### Example: Using Qwen

```python
MODEL_NAME = "Qwen/Qwen2.5-7B"

graph_config = GraphConfig(tokenizer_name=MODEL_NAME)
verifier_config = VerifierConfig(model_name=MODEL_NAME)
# ... same pattern as above
```

## Configuration

All parameters are managed via Pydantic models and support environment variables:

### GraphConfig
- `max_order`: Maximum Markov chain order (default: 5, range: 1-10)
- `tokenizer_name`: HuggingFace tokenizer (default: "openai/gpt-oss-20b") - **Must match verifier model**
- `chunk_size`: File processing chunk size (default: 10000)
- `hf_token`: HuggingFace API token for gated models (default: None)
- `download_mode`: Download acceleration - "auto", "hf_transfer", or "default" (default: "auto")

### DraftConfig
- `k`: Number of tokens to draft (default: 5)
- `strategy`: "greedy" or "sampling" (default: "greedy")
- `attentive_mix`: Enable attention-based context mixing (default: True)
- `order_bias`: β parameter for higher-order preference (default: 1.0)
- `mix_temperature`: τ parameter for attention sharpness (default: 1.0)
- `reliability_weight`: Weight for log-count reliability (default: 1.0)
- `entropy_penalty`: Penalty for distribution entropy (default: 0.5)

### VerifierConfig
- `model_name`: HuggingFace model (default: "openai/gpt-oss-20b") - **Must match graph tokenizer**
- `device`: "cuda", "cpu", or None for auto-detect
- `hf_token`: HuggingFace API token for gated models (default: None)
- `download_mode`: Download acceleration - "auto", "hf_transfer", or "default" (default: "auto")

### GenerationConfig
- `max_tokens`: Maximum tokens to generate (default: 100)
- `temperature`: Sampling temperature (default: 1.0)
- `seed`: Random seed for reproducibility (default: None)

### Environment Variables

```bash
export SPECULANT_DRAFT__K=10
export SPECULANT_DRAFT__STRATEGY=sampling
export SPECULANT_VERIFIER__MODEL_NAME=meta-llama/Llama-3.2-3B
export SPECULANT_VERIFIER__DOWNLOAD_MODE=hf_transfer
```

## Download Acceleration

Control HuggingFace model download speeds with the `download_mode` configuration:

- **`auto`** (default): Uses `hf_xet` if available (included in `huggingface_hub>=0.32.0`)
- **`hf_transfer`**: High-bandwidth optimization for cloud servers/data centers (1+ Gbps)
- **`default`**: Standard downloads without acceleration

### Installation

```bash
# For auto mode (default, recommended)
uv add huggingface_hub

# For hf_transfer mode (high-bandwidth only)
uv add "huggingface_hub[hf_transfer]"
```

### Usage

```python
# For high-bandwidth connections (cloud servers, data centers)
verifier_config = VerifierConfig(
    model_name="openai/gpt-oss-20b",
    download_mode="hf_transfer"
)

# For standard connections (default)
graph_config = GraphConfig(
    tokenizer_name="openai/gpt-oss-20b",
    download_mode="auto"
)
```

## Architecture

### Multi-Order Graph Structure

- **Nodes**: Two types:
  - Token nodes (int): Individual tokens with metadata (text, count)
  - N-gram nodes (tuple): Context sequences of length 1-5
- **Edges**: Transitions from n-gram contexts to next tokens
  - Edge attributes: `weight` (probability), `count` (frequency), `order` (context length)
- **Context Index**: O(1) lookup dictionary mapping n-grams to their order
- **Storage**: NetworkX DiGraph + context index serialized with pickle

### Adaptive Draft Generation

**Order Selection Algorithm:**
1. For each token to draft, extract last N tokens from context (N = max_order down to 1)
2. Check order-5 index → if found, draft from order-5 graph
3. If not found, check order-4, then order-3, etc.
4. Draft from highest matching order until dead-end or k tokens reached
5. When dead-end: return to step 1 with updated context

**Two Strategies:**
1. **Greedy**: Select highest probability successor from matched order
2. **Sampling**: Sample from probability distribution of matched order

**Key Advantage:** Higher-order contexts provide more accurate predictions when available, gracefully falling back to lower orders when needed.

### Verification

The verifier model uses **rejection sampling** to accept or reject draft tokens, guaranteeing that the output distribution matches what the verifier model would generate autoregressively.

**Acceptance Rule:**
- **For greedy strategy** (deterministic proposal):
  - Proposal is `q(x*) = 1` (delta function at chosen token)
  - Accept with probability: `α = P_target(x*)`
  - On rejection: sample from `P_target` conditioned on `y ≠ x*`
- **For sampling strategy** (stochastic proposal):
  - Proposal is `q` = the graph distribution at matched context
  - Accept with probability: `α = min(1, P_target(x) / q(x))`
  - On rejection: sample from residual `max(0, P_target - q)`
  - Fallback: if residual sums to 0, sample from `P_target` conditioned on `y ≠ x`

This method guarantees the output distribution is identical to autoregressive generation from the verifier.

## Server Mode (No Model Reloading!)

To avoid reloading model weights into GPU memory on every run, use the 
server mode. The server loads the model once at startup and keeps it in 
memory for fast repeated inference.

### Installation

```bash
uv sync --extra server
```

### Starting the Server

```bash
# Basic usage (uses default model)
uv run --extra server server/app.py \
  --graph-path examples/ngram_graph.pkl

# With custom model
uv run --extra server server/app.py \
  --graph-path examples/seed_ngram_graph.pkl \
  --model-name ByteDance-Seed/Seed-OSS-36B-Instruct \
  --k 8 \
  --strategy greedy \
  --host 0.0.0.0 \
  --port 8000
```

The server exposes two endpoints:
- `GET /health` - Health check with model info
- `POST /generate` - Generate text from a prompt

### Using the Client

```bash
# In another terminal, run the client example
uv run --extra server examples/example_client.py

# With custom settings
uv run --extra server examples/example_client.py \
  --url http://localhost:8000 \
  --max-tokens 100 \
  --temperature 0.9
```

### Client Code Example

```python
from examples.example_client import SpeculativeDecoderClient

client = SpeculativeDecoderClient(base_url="http://127.0.0.1:8000")

# Check health
health = client.health()
print(f"Model: {health.model_name}")

# Generate (model stays loaded between requests!)
result = client.generate(
    prompt="What is a force majeure clause?",
    max_tokens=50,
    temperature=0.9
)
print(result.text)
print(f"Acceptance rate: {result.acceptance_rate:.2%}")
```

### Benefits
- ✅ Load model once, use many times
- ✅ No GPU memory reloading between requests
- ✅ RESTful API for easy integration
- ✅ Multiple clients can connect simultaneously
- ✅ Ideal for interactive development and testing

## Example

Run the included example:

```bash
cd examples
python example.py
```

This builds a graph from legal contract text and generates responses to legal questions.

## Design Decisions

### Why Multi-Order Markov Chains?

Higher-order contexts (5-grams) capture longer-range dependencies and produce more accurate predictions when corpus patterns match the query. The adaptive algorithm automatically falls back to lower orders when high-order contexts aren't available, providing robustness.

### Why No Pruning?

Preserves the complete empirical distribution from corpus. Removing low-frequency transitions would bias sampling and break probabilistic guarantees. The sparse graph representation makes this efficient.

### Why O(1) Context Index?

With max_order=5, we could check 5 graphs sequentially. The context index allows single dictionary lookups instead, making order matching extremely fast.

### Edge Cases

- **No context matches any order**: Returns empty draft, verifier generates 1 token
- **Dead ends mid-draft**: Returns partial draft, sends to verifier
- **Prompt shorter than max_order**: Starts matching from lower orders
- **Cross-file boundaries**: N-grams can span file boundaries (corpus treated as unified)

## Visualization

Visualize the multi-order graph structure:

```bash
cd examples
python visualize_graph.py llama_knowledge_graph.pkl --max-nodes 100 --min-weight 0.1
```

- **Green nodes**: Individual tokens (order-1 contexts)
- **Orange nodes**: N-gram contexts (orders 2-5)
- **Edge width**: Proportional to transition probability
- **Hover**: See full context text and probabilities

## Requirements

- Python 3.13+
- See `pyproject.toml` for dependencies

## License

MIT
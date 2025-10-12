# Speculative Graph Decoding

Novel approach to speculative decoding using multi-order knowledge graphs as draft models instead of small LLMs.

## Overview

Traditional speculative decoding uses a small draft model to propose tokens that a large verifier model accepts or rejects. This project replaces the draft model with a **multi-order knowledge graph** built from domain-specific text corpora.

### Key Innovation

- **Multi-order Markov Chains**: Adaptively uses 1st through 5th order context for accurate predictions
- **Adaptive order selection**: Automatically selects highest available order for each prediction
- **Zero training**: No need to train or maintain a separate draft model
- **Domain-specific**: Graph captures patterns from user-supplied corpora (law, finance, healthcare, etc.)
- **Transparent**: All transitions are traceable to source text with O(1) context lookup

## Installation

```bash
uv pip install -e .
```

### Development Setup

This project uses pre-commit hooks to maintain code quality:
- **uv-lock**: Ensures the lockfile is up-to-date
- **Ruff**: Linting and code formatting
- **Conventional Commits**: Validates commit message format

Setup pre-commit hooks:

```bash
uv add --dev pre-commit
pre-commit install
pre-commit install --hook-type commit-msg
```

Run hooks manually on all files:

```bash
pre-commit run --all-files
```

## Quick Start

### 1. Build Multi-Order Knowledge Graph

```python
from speculant_graph import GraphBuilder, GraphConfig

config = GraphConfig(
    max_order=5,  # Build orders 1-5 (default)
    tokenizer_name="meta-llama/Llama-3.2-3B"
)

builder = GraphBuilder(
    tokenizer_name=config.tokenizer_name,
    max_order=config.max_order,
    chunk_size=config.chunk_size
)

graph = builder.build_from_files(["corpus1.txt", "corpus2.txt"])
builder.save("knowledge_graph.pkl")

# Graph now contains:
# - Order 1: P(next | token_i)
# - Order 2: P(next | token_i-1, token_i)
# - Order 3: P(next | token_i-2, token_i-1, token_i)
# - Order 4: P(next | token_i-3, token_i-2, token_i-1, token_i)
# - Order 5: P(next | token_i-4, ..., token_i-1, token_i)
```

### 2. Generate with Speculative Decoding

```python
from speculant_graph import (
    SpeculativeDecoder,
    DraftConfig,
    VerifierConfig,
    GenerationConfig
)

decoder = SpeculativeDecoder(
    graph_path="knowledge_graph.pkl",
    verifier_config=VerifierConfig(model_name="openai/gpt-oss-20b"),
    draft_config=DraftConfig(k=8, strategy="greedy")
)

result = decoder.generate(
    prompt="What is a force majeure clause?",
    generation_config=GenerationConfig(max_tokens=50, temperature=0.9)
)

print(result.text)
print(f"Acceptance rate: {result.acceptance_rate:.2%}")
```

## Using Different Models

**The system works with ANY HuggingFace model!** The default is `openai/gpt-oss-20b`, but you can use Llama, Mistral, Qwen, or any other model.

### Important: Tokenizer Alignment

⚠️ **Critical:** The tokenizer used to build the graph MUST match the verifier model's tokenizer. Otherwise, token IDs won't align and drafts will be meaningless.

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
    graph_path="llama_graph.pkl",
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

### VerifierConfig
- `model_name`: HuggingFace model (default: "openai/gpt-oss-20b") - **Must match graph tokenizer**
- `device`: "cuda", "cpu", or None for auto-detect
- `hf_token`: HuggingFace API token for gated models (default: None)
- `download_mode`: Download acceleration - "auto", "hf_transfer", or "default" (default: "auto")

### GenerationConfig
- `max_tokens`: Maximum tokens to generate (default: 100)
- `temperature`: Sampling temperature (default: 1.0)

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
pip install -U "huggingface_hub"

# For hf_transfer mode (high-bandwidth only)
pip install "huggingface_hub[hf_transfer]"
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

**Standard Acceptance Rule (both strategies):**
- Accept with probability: `α = min(1, P_target(x) / q(x))`
  - Where `q` is the full draft distribution over successors at the matched context
  - For greedy: `q` is concentrated on the argmax token
  - For sampling: `q` is the distribution the token was sampled from
- On rejection: sample correction from residual `max(0, P_target - q)`
  - Builds sparse `q` vector from all successors at the matched context
  - Computes per-token residual: `residual(x) = max(0, p(x) - q(x))`
  - Renormalizes and samples
- Fallback: if residual sums to 0, sample from `P_target` conditioned on `y ≠ x`

This method guarantees the output distribution is identical to autoregressive generation from the verifier.

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
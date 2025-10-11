# Speculative Graph Decoding

Novel approach to speculative decoding using knowledge graphs as draft models instead of small LLMs.

## Overview

Traditional speculative decoding uses a small draft model to propose tokens that a large verifier model accepts or rejects. This project replaces the draft model with a **knowledge graph** built from domain-specific text corpora.

### Key Innovation

- **Graph-based drafting**: First-order Markov Chain represents token transitions from corpus
- **Zero training**: No need to train or maintain a separate draft model
- **Domain-specific**: Graph captures patterns from user-supplied corpora (law, finance, healthcare, etc.)
- **Transparent**: All transitions are traceable to source text

## Installation

```bash
uv pip install -e .
```

## Quick Start

### 1. Build Knowledge Graph

```python
from speculant_graph import GraphBuilder, GraphConfig

config = GraphConfig()
builder = GraphBuilder(
    tokenizer_name=config.tokenizer_name,
    chunk_size=config.chunk_size
)

graph = builder.build_from_files(["corpus1.txt", "corpus2.txt"])
builder.save("knowledge_graph.pkl")
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
    verifier_config=VerifierConfig(acceptance_threshold=0.6),
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
    model_name=MODEL_NAME,
    acceptance_threshold=0.6
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
- `tokenizer_name`: HuggingFace tokenizer (default: "openai/gpt-oss-20b") - **Must match verifier model**
- `chunk_size`: File processing chunk size (default: 10000)

### DraftConfig
- `k`: Number of tokens to draft (default: 5)
- `strategy`: "greedy" or "sampling" (default: "greedy")

### VerifierConfig
- `model_name`: HuggingFace model (default: "openai/gpt-oss-20b") - **Must match graph tokenizer**
- `device`: "cuda", "cpu", or None for auto-detect
- `acceptance_threshold`: Probability threshold for accepting drafts (default: 0.5)

### GenerationConfig
- `max_tokens`: Maximum tokens to generate (default: 100)
- `temperature`: Sampling temperature (default: 1.0)

### Environment Variables

```bash
export SPECULANT_DRAFT__K=10
export SPECULANT_DRAFT__STRATEGY=sampling
export SPECULANT_VERIFIER__ACCEPTANCE_THRESHOLD=0.7
```

## Architecture

### Graph Structure

- **Nodes**: Token IDs with metadata (text, count)
- **Edges**: Transitions with probabilities P(j|i) = count(i→j) / count(i)
- **Storage**: NetworkX DiGraph serialized with pickle

### Draft Generation

Two strategies:
1. **Greedy**: Select highest probability successor at each step
2. **Sampling**: Sample from transition probability distribution

### Verification

The verifier model (gpt-oss-20b) accepts or rejects draft tokens based on its own probability distribution and the configured acceptance threshold.

## Example

Run the included example:

```bash
cd examples
python example.py
```

This builds a graph from legal contract text and generates responses to legal questions.

## Design Decisions

### Why First-Order Markov Chains?

Starting simple with single-token conditioning. Architecture supports future n-gram extensions (bigrams, trigrams).

### Why No Pruning?

Preserves the complete empirical distribution from corpus. Removing low-frequency transitions would bias sampling and break probabilistic guarantees.

### Edge Cases

- **Unknown starting token**: Falls back to most frequent token in graph
- **Dead ends**: Returns partial draft (< k tokens)
- **Empty corpus**: Works with any size corpus, no minimum validation

## Requirements

- Python 3.14+
- See `pyproject.toml` for dependencies

## License

MIT
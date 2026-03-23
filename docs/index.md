# Speculant Graph

**Graph drafts, LLM verifies** — a novel speculative decoding framework using multi-order n-gram graphs.

## What is this?

Traditional speculative decoding uses a small draft model to propose tokens that a large verifier model accepts or rejects. Speculant Graph replaces the draft model with a **multi-order n-gram graph** built from domain-specific text corpora.

## Key Features

- **Multi-order Markov Chains**: Adaptively uses 1st through 5th order context for accurate predictions
- **Attentive Context Mixing**: Attention mechanism that blends multiple n-gram orders
- **Zero training**: No need to train or maintain a separate draft model
- **Domain-specific**: Graph captures patterns from user-supplied corpora
- **2-5x speedup**: Over standard autoregressive decoding

## Quick Install

```bash
uv sync
```

## Quick Example

```python
from speculant_graph import (
    GraphBuilder, SpeculativeDecoder,
    DraftConfig, VerifierConfig, GenerationConfig
)

# Build graph
builder = GraphBuilder(tokenizer_name="your-model", max_order=5)
builder.build_from_files(["corpus.txt"])
builder.save("graph.pkl")

# Generate with speculative decoding
decoder = SpeculativeDecoder(
    graph_path="graph.pkl",
    verifier_config=VerifierConfig(model_name="your-model"),
    draft_config=DraftConfig(k=8, strategy="greedy"),
)
result = decoder.generate("Your prompt", GenerationConfig(max_tokens=100))
print(result.text)
```

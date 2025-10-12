# Product Requirements Document (PRD)
## Speculative Decoding with N-gram Graph Drafter

---

## 1. Executive Summary

### 1.1 Overview
This project implements a novel approach to speculative decoding where the draft model is replaced by a multi-order n-gram graph built from domain-specific text corpora. Instead of using a small LLM as the drafter, we use Markov Chains of varying orders (1st through 5th order by default) to capture token transition patterns extracted from user-supplied text files.

### 1.2 Key Innovation
- **Traditional speculative decoding:** Small draft model → Large verifier model
- **Our approach:** N-gram graph drafter → Large verifier model

### 1.3 Benefits
- No need to train/maintain a separate draft model
- Domain-specific: Graph built from relevant corpus reduces hallucinations
- Transparent: Graph transitions are interpretable and traceable to source text
- Efficient: Fast graph traversal for draft generation

---

## 2. System Architecture

### 2.1 High-Level Components

```
┌─────────────────┐
│  Text Corpus    │
│  (.txt files)   │
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│  Graph Constructor  │
│  - Tokenization     │
│  - Transition Count │
│  - Probability Calc │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  N-gram Graph       │
│  (NetworkX)         │
│  - Nodes: n-grams   │
│  - Edges: P(j|ctx)  │
│  - Orders: 1-5      │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Draft Generator   │
│   - Multi-order     │
│   - Greedy/Sampling │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Verifier Model     │
│  (gpt-oss-20b)      │
│  - Accept/Reject    │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Final Output      │
└─────────────────────┘
```

### 2.2 Technology Stack
- **Language:** Python 3.13
- **Graph Library:** NetworkX 3.5
- **Tokenizer:** HuggingFace Transformers 4.57.0
- **ML Framework:** PyTorch 2.8.0
- **Data Validation:** Pydantic 2.12.0
- **Logging:** Loguru 0.7.3
- **Package Manager:** uv

---

## 3. Detailed Requirements

### 3.1 Graph Construction

#### 3.1.1 Input
- **Corpus format:** Plain text (.txt) files
- **Corpus size:** Scalable from small (few KB) to large (books, 100MB+)
- **Multiple files:** Support processing multiple .txt files as single corpus

#### 3.1.2 Tokenization
- **Tokenizer:** Must use `openai/gpt-oss-20b` tokenizer (consistency with verifier)
- **Special tokens:** Include BOS, EOS, and other special tokens in graph
- **Metadata:** Store tokenizer name and version with graph for validation

#### 3.1.3 Graph Structure
- **Type:** Directed, weighted graph (NetworkX DiGraph)
- **Nodes:** Two types:
  - **Token nodes (int):** Individual tokens with count attribute
  - **N-gram nodes (tuple):** Contexts of length 1-5 (e.g., `(token_i, token_j)`)
- **Node attributes:**
  - For token nodes: `token_id`, `text`, `count`
  - For n-gram nodes: No attributes (exist only as edge sources)
- **Edges:** Transitions from n-gram context to next token
- **Edge attributes:**
  - `weight`: Transition probability P(next | context)
  - `count`: Frequency count of transition
  - `order`: Length of the source context (1-5)

#### 3.1.4 Probability Calculation
- **Method:** Simple frequency counts for each order
  ```
  For order-n:
  P(token_next | context_n) = count(context_n → next) / Σ count(context_n → *)

  Example order-3:
  P(token_k | token_i, token_j) = count((i,j) → k) / Σ count((i,j) → *)
  ```
- **No smoothing:** Unseen transitions have probability 0
- **No pruning:** Keep ALL observed transitions, regardless of frequency

#### 3.1.5 Multi-Order Markov Chains
- **Default max_order:** 5 (configurable 1-10)
- **Orders built:** All orders from 1 to max_order simultaneously
  - Order 1: P(next | token_i)
  - Order 2: P(next | token_i-1, token_i)
  - Order 3: P(next | token_i-2, token_i-1, token_i)
  - Order 4: P(next | token_i-3, token_i-2, token_i-1, token_i)
  - Order 5: P(next | token_i-4, token_i-3, token_i-2, token_i-1, token_i)
- **Context Index:** Fast O(1) lookup dictionary mapping n-grams to their order

#### 3.1.6 Multi-File Handling
- **Cross-boundary transitions:** Track transitions across file boundaries for all orders
- **Rationale:** User provides multiple files as single corpus, intent is unified graph

#### 3.1.7 Memory Optimization
- **Streaming processing:** Process large files in chunks to avoid memory overflow
- **Sparse representation:** Only store observed transitions (NetworkX default)

#### 3.1.8 Serialization
- **Save format:** Pickle (includes graph + context_index + metadata)
- **Include metadata:**
  - Tokenizer config
  - max_order
  - Corpus stats (nodes/edges per order)
  - Build timestamp
- **Load validation:** Verify tokenizer compatibility on load

---

### 3.2 Draft Generation

#### 3.2.1 Input
- **User prompt:** Text string
- **Draft length k:** Number of sequential tokens to generate
- **Strategy:** Greedy top-k OR sampling

#### 3.2.2 Multi-Order Context Matching
- **Algorithm:** For each token to draft:
  1. Extract last N tokens from current context (N = max_order down to 1)
  2. Check order-5 context index → if found, use order-5 graph
  3. If not found, check order-4 → use order-4 graph
  4. Continue decreasing order until match found
  5. If no match at any order → return empty draft (verifier generates)

#### 3.2.3 Draft Strategies

**Strategy 1: Greedy**
- Select highest probability successor from matched order
- Continue drafting from same order until dead-end
- When dead-end hit, send drafted tokens to verifier
- On next iteration, re-match highest order with new context

**Strategy 2: Sampling**
- Sample from probability distribution of matched order
- Continue drafting from same order until dead-end
- When dead-end hit, send drafted tokens to verifier
- On next iteration, re-match highest order with new context

#### 3.2.4 Multi-Order Traversal Logic
```
draft_tokens = []
current_context = prompt_tokens

while len(draft_tokens) < k:
    # Find highest matching order
    for order in [5, 4, 3, 2, 1]:
        context_ngram = tuple(current_context[-order:])
        if context_ngram in context_index:
            # Draft from this order until dead-end
            while len(draft_tokens) < k:
                successors = graph.successors(context_ngram)
                if no successors:
                    break

                next_token = select(successors, strategy)  # greedy or sampling
                draft_tokens.append(next_token)

                # Update context (sliding window)
                current_context.append(next_token)
                context_ngram = tuple(current_context[-order:])

                if context_ngram not in context_index:
                    break  # Drop to lower order

            break  # Exit order search loop

    if no order matched:
        break  # Return partial draft

return draft_tokens
```

#### 3.2.5 Edge Cases

| Case | Behavior |
|------|----------|
| No context matches any order | Return empty draft, verifier generates 1 token |
| Dead end (no successors) | Return partial draft (< k tokens) |
| Context found in order-N | Draft from order-N, may drop to order-(N-1) mid-draft |
| Prompt shorter than max_order | Start matching from lower orders |
| k > available path | Return what's possible |

#### 3.2.6 Output Format
- **Type:** List of token IDs
- **Length:** Up to k tokens (may be less)
- **Metadata:** Strategy used, actual length, whether terminated early

---

### 3.3 Verification & Inference

#### 3.3.1 Verifier Model
- **Model:** `openai/gpt-oss-20b` from HuggingFace
- **Loading:** Use HuggingFace Transformers library
- **Tokenizer:** Same as graph construction

#### 3.3.2 Rejection Sampling
- **Input:** User prompt + draft tokens + proposal metadata per step
- **Algorithm:** For each draft token (verified in parallel via single forward pass):

  **For greedy strategy (deterministic proposal):**
  1. Proposal distribution: `q(x*) = 1` (delta function at argmax token)
  2. Accept with probability: `α = P_target(x*)`
  3. If rejected: sample from `P_target` conditioned on `y ≠ x*`
     - Set `P(x*) = 0`, renormalize, sample
  4. Stop at first rejection

  **For sampling strategy (stochastic proposal):**
  1. Proposal distribution: `q` = the graph distribution at matched context
  2. Accept with probability: `α = min(1, P_target(x) / q(x))`
  3. If rejected: sample from residual `max(0, P_target - q)`
     - Build sparse q vector from full successor distribution at matched context
     - Compute residual per-token: `residual(x) = max(0, p(x) - q(x))`
     - Renormalize and sample
  4. Fallback: if residual sums to 0, sample from `P_target` conditioned on `y ≠ x`
  5. Stop at first rejection

- **Guarantee:** Output distribution is identical to autoregressive generation from verifier
- **Output:** Accepted tokens + optional correction token

#### 3.3.3 End-to-End Pipeline
```
1. User provides prompt
2. Draft generator produces k tokens from graph
3. Verifier evaluates draft
4. Return final output (accepted + generated)
```

---

## 4. Functional Requirements

### 4.1 Must Have (Phase 1)
- ✅ Parse .txt files and build graph
- ✅ Multi-order Markov chains (orders 1-5)
- ✅ Support both greedy and sampling draft strategies
- ✅ Generate k sequential draft tokens with adaptive order selection
- ✅ Integrate with HuggingFace verifier models
- ✅ Handle special tokens (BOS, EOS)
- ✅ Save/load graph from disk with context index
- ✅ End-to-end runnable system
- ✅ Graph visualization tools

### 4.2 Should Have (Phase 2)
- ⏳ Additional file formats (.json, .md, .pdf)
- ⏳ Performance benchmarking vs baseline
- ⏳ Configurable max_order beyond 5

### 4.3 Could Have (Future)
- ⏳ Incremental graph updates (add new docs without rebuild)
- ⏳ Graph pruning options (user-configurable)
- ⏳ Alternative verifier models
- ⏳ Web interface for graph exploration

---

## 5. Non-Functional Requirements

### 5.1 Performance
- **Graph construction:** Handle 100MB+ text files
- **Inference latency:** Draft generation < 50ms for k=10
- **Memory:** Efficient for vocabularies up to 100K tokens

### 5.2 Scalability
- **Corpus size:** Support gigabyte-scale corpora
- **Graph size:** Handle millions of edges

### 5.3 Reliability
- **Validation:** Check tokenizer compatibility on load
- **Error handling:** Graceful failures for edge cases
- **Determinism:** Greedy mode produces reproducible results

### 5.4 Usability
- **API:** Simple Python interface
- **Documentation:** Clear examples and docstrings
- **Configurability:** Easy to adjust k, strategy, model paths

---

## 6. System Interface

### 6.1 Graph Builder API
```python
from speculant_graph import GraphBuilder, GraphConfig

# Build multi-order graph from corpus
graph_config = GraphConfig(
    max_order=5,
    tokenizer_name="meta-llama/Llama-3.2-3B",
    hf_token="your_token"
)

builder = GraphBuilder(
    tokenizer_name=graph_config.tokenizer_name,
    max_order=graph_config.max_order,
    chunk_size=graph_config.chunk_size,
    hf_token=graph_config.hf_token
)

graph = builder.build_from_files(["corpus1.txt", "corpus2.txt"])
builder.save("knowledge_graph.pkl")
```

### 6.2 Draft Generator API
```python
from speculant_graph import DraftGenerator

# Load graph
generator = DraftGenerator.from_file("knowledge_graph.pkl")

# Generate draft
draft = generator.generate(
    prompt="What is a force majeure clause?",
    k=10,
    strategy="greedy"  # or "sampling"
)
```

### 6.3 End-to-End Inference API
```python
from speculant_graph import (
    SpeculativeDecoder,
    VerifierConfig,
    DraftConfig,
    GenerationConfig
)

# Configure
verifier_config = VerifierConfig(
    model_name="meta-llama/Llama-3.2-3B",
    hf_token="your_token"
)
draft_config = DraftConfig(k=8, strategy="greedy")

# Initialize decoder
decoder = SpeculativeDecoder(
    graph_path="knowledge_graph.pkl",
    verifier_config=verifier_config,
    draft_config=draft_config
)

# Generate with multi-order speculative decoding
generation_config = GenerationConfig(max_tokens=50, temperature=0.9)
result = decoder.generate(
    prompt="A termination clause outlines the circumstances under which",
    generation_config=generation_config
)

print(result.text)
print(f"Acceptance rate: {result.acceptance_rate:.2%}")
print(f"Breakdown: {result.num_accepted} accepted, {result.num_rejected} rejected")
```

---

## 7. Testing Requirements

### 7.1 Unit Tests
- Tokenization correctness
- Graph construction (node/edge counts)
- Probability calculation accuracy
- Draft generation (greedy vs sampling)
- Edge case handling

### 7.2 Integration Tests
- End-to-end pipeline
- Save/load graph
- Multi-file corpus processing

### 7.3 Performance Tests
- Large corpus handling (100MB+)
- Draft generation latency
- Memory usage profiling

---

## 8. Success Metrics

### 8.1 Correctness
- ✅ Graph accurately represents corpus transitions
- ✅ Probabilities sum to 1.0 for each node
- ✅ Special tokens handled correctly

### 8.2 Performance
- 📊 Draft acceptance rate > baseline (measure against no drafting)
- 📊 End-to-end speedup > 1.0x
- 📊 Memory usage acceptable for target corpus sizes

### 8.3 Quality
- 📊 Reduced hallucinations (stays within corpus knowledge)
- 📊 Domain-relevant completions

---

## 9. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Low acceptance rate | High | Use n-grams, tune k parameter |
| Graph too large | Medium | Sparse representation, efficient serialization |
| Unknown tokens common | Medium | Better fallback strategies |
| Corpus-query mismatch | High | Documentation on corpus selection |

---

## 10. Timeline & Milestones

### Phase 1: Core Implementation (Week 1-2)
- ✅ Project setup
- ✅ Graph construction
- ✅ Draft generation
- ✅ Basic verification

### Phase 2: Integration & Testing (Week 3)
- ✅ End-to-end pipeline
- ✅ Unit tests
- ✅ Example scripts

### Phase 3: Evaluation (Week 4)
- 📊 Performance benchmarking
- 📊 Quality evaluation
- 📝 Documentation

---

## 11. Open Questions

1. **Optimal k value:** What draft length maximizes speedup? (Experiment-driven)
2. **Temperature for sampling:** Should we add temperature parameter? (Phase 2)
3. **Hybrid strategies:** Combine greedy + sampling? (Phase 2)
4. **Graph compression:** Can we reduce memory without losing quality? (Future)

---

## 12. Appendix

### 12.1 Speculative Decoding Background
Speculative decoding generates draft tokens cheaply (small model) and verifies in parallel with large model. Accepted tokens provide speedup. Our innovation: replace small model with multi-order n-gram graph.

### 12.2 Markov Chain Terminology
- **Order-1 (unigram context):** P(token_n | token_{n-1})
- **Order-2 (bigram context):** P(token_n | token_{n-2}, token_{n-1})
- **Order-3 (trigram context):** P(token_n | token_{n-3}, token_{n-2}, token_{n-1})
- **Order-N (n-gram context):** P(token_n | token_{n-N}, ..., token_{n-1})
- **Adaptive order selection:** Choose highest available order for each prediction

### 12.3 References
- Speculative Decoding: [Original Paper]
- NetworkX Documentation: https://networkx.org/
- HuggingFace Transformers: https://huggingface.co/docs/transformers/

---

**Document Version:** 2.0
**Last Updated:** 2025-10-12
**Author:** Product Specification for Speculative Graph Decoding System
**Major Changes in v2.0:**
- Implemented multi-order Markov chains (orders 1-5)
- Added adaptive order selection algorithm
- Context index for O(1) n-gram lookups
- Updated all code examples to reflect new architecture

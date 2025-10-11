# Product Requirements Document (PRD)
## Speculative Decoding with Knowledge Graph Drafter

---

## 1. Executive Summary

### 1.1 Overview
This project implements a novel approach to speculative decoding where the draft model is replaced by a knowledge graph built from domain-specific text corpora. Instead of using a small LLM as the drafter, we use a first-order Markov Chain representation of token transitions extracted from user-supplied text files.

### 1.2 Key Innovation
- **Traditional speculative decoding:** Small draft model → Large verifier model
- **Our approach:** Knowledge graph drafter → Large verifier model

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
│  Knowledge Graph    │
│  (NetworkX)         │
│  - Nodes: Tokens    │
│  - Edges: P(j|i)    │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Draft Generator   │
│   - Greedy top-k    │
│   - Sampling        │
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
- **Language:** Python 3.14
- **Graph Library:** NetworkX
- **Tokenizer:** HuggingFace Transformers (`openai/gpt-oss-20b`)
- **Model:** `openai/gpt-oss-20b` (verifier)
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
- **Nodes:** Individual tokens (token IDs from tokenizer)
- **Node attributes:**
  - `token_id`: Integer token ID
  - `text`: Human-readable token text (for debugging)
  - `count`: Total occurrences in corpus
- **Edges:** Transitions from token_i to token_j
- **Edge attributes:**
  - `weight`: Transition probability P(j|i)
  - `count`: Frequency count of transition

#### 3.1.4 Probability Calculation
- **Method:** Simple frequency counts
  ```
  P(token_j | token_i) = count(i → j) / count(i)
  ```
- **No smoothing:** Unseen transitions have probability 0
- **No pruning:** Keep ALL observed transitions, regardless of frequency

#### 3.1.5 Markov Order
- **Initial implementation:** First-order (single token nodes)
- **Architecture requirement:** Extensible to n-grams (bigrams, trigrams) in future

#### 3.1.6 Multi-File Handling
- **Cross-boundary transitions:** Track transitions across file boundaries
- **Rationale:** User provides multiple files as single corpus, intent is unified graph

#### 3.1.7 Memory Optimization
- **Streaming processing:** Process large files in chunks to avoid memory overflow
- **Sparse representation:** Only store observed transitions (NetworkX default)

#### 3.1.8 Serialization
- **Save format:** Pickle or GraphML
- **Include metadata:** Tokenizer config, corpus stats, build timestamp
- **Load validation:** Verify tokenizer compatibility on load

---

### 3.2 Draft Generation

#### 3.2.1 Input
- **User prompt:** Text string
- **Draft length k:** Number of sequential tokens to generate
- **Strategy:** Greedy top-k OR sampling

#### 3.2.2 Starting Point
- **Primary:** Last token of user prompt
- **Fallback:** If last token not in graph, use most frequent token in graph

#### 3.2.3 Draft Strategies

**Strategy 1: Greedy Top-K**
- At each step, select highest probability successor
- Deterministic output
- Generate k sequential tokens

**Strategy 2: Probability-Based Sampling**
- At each step, sample from successor probability distribution
- Stochastic output
- Generate k sequential tokens

#### 3.2.4 Traversal Logic
```
1. Start from last token of prompt (or fallback)
2. For i = 1 to k:
   a. Get all successor nodes and their probabilities
   b. Select next token (greedy or sampling)
   c. Append to draft sequence
   d. If no successors OR hit EOS: break (return partial draft)
3. Return draft sequence (may be < k tokens)
```

#### 3.2.5 Edge Cases

| Case | Behavior |
|------|----------|
| Unknown starting token | Use most frequent token in graph |
| Dead end (no successors) | Return partial draft (< k tokens) |
| Hit EOS token | Stop early, return partial draft |
| k > available path | Return what's possible |
| Very low probability path | Trust the sampling |

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

#### 3.3.2 Acceptance Sampling
- **Input:** User prompt + draft tokens
- **Process:**
  1. Verifier generates logits for each position
  2. Compare draft tokens with verifier distribution
  3. Accept/reject based on acceptance sampling algorithm
- **Output:** Accepted prefix + new tokens from verifier

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
- ✅ First-order Markov chain (single token nodes)
- ✅ Support both greedy and sampling draft strategies
- ✅ Generate k sequential draft tokens
- ✅ Integrate with `openai/gpt-oss-20b` verifier
- ✅ Handle special tokens (BOS, EOS)
- ✅ Save/load graph from disk
- ✅ End-to-end runnable system

### 4.2 Should Have (Phase 2)
- ⏳ Support for n-gram contexts (bigrams, trigrams)
- ⏳ Additional file formats (.json, .md, .pdf)
- ⏳ Graph visualization tools
- ⏳ Performance benchmarking vs baseline

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
from speculant_graph import GraphBuilder

# Build graph from corpus
builder = GraphBuilder(
    tokenizer_name="openai/gpt-oss-20b",
    corpus_files=["corpus1.txt", "corpus2.txt"]
)
graph = builder.build()

# Save graph
graph.save("knowledge_graph.pkl")
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
from speculant_graph import SpeculativeDecoder

# Initialize
decoder = SpeculativeDecoder(
    graph_path="knowledge_graph.pkl",
    model_name="openai/gpt-oss-20b"
)

# Generate with speculative decoding
output = decoder.generate(
    prompt="What is a force majeure clause?",
    draft_k=10,
    max_tokens=100,
    strategy="greedy"
)

print(output.text)
print(f"Acceptance rate: {output.acceptance_rate}")
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
Speculative decoding generates draft tokens cheaply (small model) and verifies in parallel with large model. Accepted tokens provide speedup. Our innovation: replace small model with knowledge graph.

### 12.2 Markov Chain Terminology
- **First-order:** P(token_n | token_{n-1})
- **Zero-order:** P(token_n) - no context
- **N-gram:** P(token_n | token_{n-N+1}, ..., token_{n-1})

### 12.3 References
- Speculative Decoding: [Original Paper]
- NetworkX Documentation: https://networkx.org/
- HuggingFace Transformers: https://huggingface.co/docs/transformers/

---

**Document Version:** 1.0
**Last Updated:** 2025-10-11
**Author:** Product Specification for Speculative Graph Decoding System

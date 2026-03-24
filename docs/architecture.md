# Architecture

## Multi-Order Graph Structure

- **Token nodes** (`int`): Individual tokens with metadata (text, count)
- **N-gram nodes** (`tuple`): Context sequences of length 1-5
- **Edges**: Transitions from n-gram contexts to next tokens with `weight` (probability), `count` (frequency), `order` (context length)
- **Context Index**: O(1) lookup dictionary mapping n-grams to their order

## Adaptive Draft Generation

1. Extract last N tokens from context (N = max_order down to 1)
2. Check order-5 index → if found, draft from order-5 graph
3. If not found, fall back to order-4, then order-3, etc.
4. Draft from highest matching order until dead-end or k tokens reached

### Strategies

- **Greedy**: Select highest probability successor
- **Sampling**: Sample from probability distribution

### Attentive Context Mixing

When enabled (default), blends multiple order contexts with attention weights:

1. Find all matching orders
2. Compute score: `β·log(o) + λ·log(count+1) - α·H` for each
3. Apply softmax to get attention weights
4. Weighted average of all distributions

## Verification

Uses **rejection sampling** to accept or reject draft tokens:

- **Greedy**: Accept with probability `P_target(x*)`
- **Sampling**: Accept with probability `min(1, P_target(x) / q(x))`

On rejection, sample a correction token from the residual distribution. This guarantees the output matches autoregressive generation from the verifier.

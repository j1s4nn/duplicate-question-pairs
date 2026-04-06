# Architecture Notes — DuplicateIQ

## Why Siamese Neural Networks?

A Siamese network uses *two identical sub-networks* sharing the same weights to process a pair of inputs independently, then compares their output representations.

For duplicate detection:
- **Left network**: Encodes Question A → Embedding vector
- **Right network**: Encodes Question B → Embedding vector  
- **Comparison**: Cosine similarity between the two vectors

The key enterprise advantage: embeddings for existing questions can be **pre-computed and stored**. When a new question arrives, only one forward pass is needed — then we compare against the stored index instantly.

## Threshold Tuning

The threshold is the decision boundary for `is_duplicate`.

- **Too low** → False positives. Different questions get merged. Users lose trust.
- **Too high** → False negatives. Duplicates slip through. DB gets bloated.

Default: `0.85` — validated as a good balance for English FAQ content.

**Precision > Recall** is our design philosophy:
- It is better to surface a duplicate for manual review than to auto-merge incorrectly.

## Hard Negatives

Questions that *look* similar syntactically but have different intents:
- "How to **start** a car?" vs "How to **stop** a car?"
- "Can I **add** a user?" vs "Can I **remove** a user?"

These are handled by SBERT's contextual embeddings (which attend to every word in context), unlike TF-IDF which only sees word frequency.

## Text Normalization Pipeline

1. Unicode normalization (NFKC form)
2. Contraction expansion (`don't` → `do not`)
3. Whitespace normalization
4. Note: Stop words like "not" are *retained* — they carry semantic intent

## Embedding Cache

Frequently asked questions generate the same embedding. We maintain an in-memory LRU-style cache keyed on the raw question string to avoid redundant model inference.

## FAISS Index Strategy

We use `IndexFlatIP` (inner product) with L2-normalized embeddings, which is mathematically equivalent to cosine similarity. For production scale (millions of questions), upgrade to `IndexIVFFlat` or `IndexHNSWFlat` for approximate nearest neighbor search with better performance.

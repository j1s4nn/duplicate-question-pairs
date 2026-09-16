# DuplicateIQ — Semantic Duplicate Question Detection

> Detects duplicate questions by **meaning**, not by matching words — Sentence-BERT embeddings + FAISS vector search behind a FastAPI service.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)
![Sentence-BERT](https://img.shields.io/badge/SBERT-all--MiniLM--L6--v2-FFD21E?logo=huggingface&logoColor=black)
![FAISS](https://img.shields.io/badge/FAISS-IndexFlatIP-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

DuplicateIQ is a self-hosted NLP service that decides whether two questions are semantically equivalent — *"How do I reset my password?"* vs *"I forgot my login credentials, what should I do?"* — and can search an index of known questions for near-duplicates of a new one. It exposes single-pair, batch, and index-management endpoints through FastAPI, with a browser frontend and an optional embeddable widget.

## Motivation / Problem

FAQ pages, support desks, and Q&A platforms accumulate restated questions constantly. Lexical matching (SQL `LIKE`, keyword overlap) fails on paraphrases: the words differ while the intent is identical. Manual moderation does not scale. The goal here is a low-latency semantic layer that flags duplicates at ingest time, with a **precision-first** policy — wrongly merging two distinct questions is far more damaging than missing an occasional duplicate.

## Methodology

```
┌─────────────────────────────────────────────────────────────┐
│                     User Query / API Call                   │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                 FastAPI Backend (Port 8000)                  │
│  ┌─────────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │  /detect    │  │  /detect/batch   │  │  /search      │  │
│  │  POST       │  │  POST (≤50)      │  │  POST         │  │
│  └──────┬──────┘  └────────┬─────────┘  └──────┬────────┘  │
│         └─────────────────┬┘                   │           │
│                      ┌────▼────────────────────▼────────┐  │
│                      │    DuplicateDetectionEngine       │  │
│                      │  ┌─────────────────────────────┐  │  │
│                      │  │  TextNormalizer              │  │  │
│                      │  │  • Contraction expansion     │  │  │
│                      │  │  • Unicode normalization     │  │  │
│                      │  └─────────┬───────────────────┘  │  │
│                      │            ▼                       │  │
│                      │  ┌─────────────────────────────┐  │  │
│                      │  │  Sentence-BERT (SBERT)       │  │  │
│                      │  │  Model: all-MiniLM-L6-v2    │  │  │
│                      │  │  Embedding dim: 384          │  │  │
│                      │  └─────────┬───────────────────┘  │  │
│                      │            ▼                       │  │
│                      │  ┌─────────────────────────────┐  │  │
│                      │  │  FAISS Index (IndexFlatIP)   │  │  │
│                      │  │  L2-normalized cosine search │  │  │
│                      │  │  Sub-millisecond at scale    │  │  │
│                      │  └─────────────────────────────┘  │  │
│                      └───────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                      │
         ┌────────────┴─────────────┐
         ▼                          ▼
┌─────────────────┐      ┌─────────────────────────┐
│  Frontend UI    │      │  Embeddable Widget      │
│  (Port 8000     │      │  converge_widget_       │
│   /frontend)    │      │  port8007.html          │
└─────────────────┘      └─────────────────────────┘
```

Pipeline per request:

1. **Normalize** — contraction expansion ("don't" → "do not") and Unicode normalization, so surface noise does not perturb embeddings.
2. **Embed** — Sentence-BERT (`all-MiniLM-L6-v2`) maps each question to a 384-dim vector. Unlike plain BERT pairwise encoding, SBERT embeddings are computed once per question and cached, making index lookup O(1) per new question.
3. **Search / compare** — vectors are L2-normalized and compared by inner product in a FAISS `IndexFlatIP` index, which is exact cosine similarity with sub-millisecond retrieval.
4. **Decide** — similarity is compared against a configurable threshold (default 0.85) and returned with a confidence band and measured latency.

## Model & Algorithm

| Component | Choice | Rationale |
|---|---|---|
| Encoder | `all-MiniLM-L6-v2` (SBERT) | Best speed/accuracy balance for short-question embeddings; 384-dim keeps the index small |
| Similarity | Cosine (normalized inner product) | Directional similarity, unaffected by embedding magnitude — the right geometry for semantic matching |
| Index | FAISS `IndexFlatIP` | Exact search at FAQ scale; swappable to IVF/HNSW for millions of vectors |
| Policy | Precision-first threshold | False positives (merging distinct questions) destroy user trust; missed duplicates are recoverable |
| Fallback | TF-IDF + cosine (scikit-learn) | Keeps the identical API contract in environments without torch/FAISS (CI, lightweight deploys) |

## API Reference

### `POST /api/detect`
Compare a single question pair.

```json
// Request
{
  "question_a": "How do I reset my password?",
  "question_b": "I forgot my login credentials, what should I do?",
  "threshold": 0.85
}

// Response
{
  "question_a": "How do I reset my password?",
  "question_b": "I forgot my login credentials, what should I do?",
  "similarity_score": 0.912,
  "is_duplicate": true,
  "threshold_used": 0.85,
  "confidence": "HIGH",
  "latency_ms": 38.4
}
```

### `POST /api/detect/batch`
Process up to 50 pairs at once.

```json
// Request
{
  "pairs": [
    { "question_a": "...", "question_b": "...", "threshold": 0.85 }
  ]
}
```

### `POST /api/search`
Find similar questions in the index.

```json
{ "question": "How to cancel?", "top_k": 5, "threshold": 0.75 }
```

### `POST /api/index/question`
Add a question to the vector index.

```json
{ "question": "How do I reset my password?", "id": "optional-custom-id" }
```

### `GET /api/stats` — Usage statistics
### `GET /health` — Health check
### `DELETE /api/index/reset` — Reset index

Full interactive docs: `http://localhost:8000/api/docs`

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the API

```bash
python run.py
# API running at http://localhost:8000
# Docs at http://localhost:8000/api/docs
```

### 3. Open the frontend

Open `frontend/index.html` in your browser (or serve it):

```bash
cd frontend && python -m http.server 3000
# Visit http://localhost:3000
```

### 4. Optional: embeddable widget

`converge_widget_port8007.html` is a standalone widget built for **Converge**, a companion media-tools web project (in development, not yet published). It can be served independently and embedded in any page as an iframe:

```bash
python -m http.server 8007
```

## Running Tests

```bash
pytest tests/ -v
```

## Configuration

| Parameter | Default | Description |
|---|---|---|
| Model | `all-MiniLM-L6-v2` | SBERT model (balances speed + accuracy) |
| Threshold | `0.85` | Cosine similarity cutoff |
| Max batch | `50` | Pairs per batch request |
| Target latency | `<100ms` | Per-request inference target |

**Better accuracy** (slower): use `all-mpnet-base-v2` in `engine.py`
**Faster** (less accurate): use `all-MiniLM-L12-v2`

## Fallback Mode

If `sentence-transformers` / FAISS are unavailable (e.g., CI/CD, lightweight environments), the engine automatically falls back to **TF-IDF + cosine similarity** (scikit-learn). All API contracts remain identical.

## Deployment

```bash
# Production with Gunicorn
pip install gunicorn
gunicorn backend.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## Tech Stack

`Python` · `Sentence-BERT (all-MiniLM-L6-v2)` · `FAISS` · `FastAPI` · `Uvicorn` · `Pydantic` · `scikit-learn (fallback)` · `pytest`

## Future Improvements

- Evaluate on a labeled duplicate-question benchmark (e.g., Quora Question Pairs) and report precision/recall at operating thresholds instead of design targets
- Cross-encoder re-ranking (e.g., `cross-encoder/stsb-distilroberta-base`) for borderline pairs above a first-stage cutoff
- Approximate indices (IVF-PQ / HNSW) and embedding caching for million-scale question corpora
- Multilingual encoder (`paraphrase-multilingual-MiniLM`) for non-English support
- Persistence layer for the index (currently in-memory) and Docker packaging

## Project Structure

```
duplicate-question-pairs/
├── README.md
├── requirements.txt
├── run.py                            ← Start the API server
├── backend/
│   ├── main.py                       ← FastAPI routes + middleware
│   ├── engine.py                     ← SBERT + FAISS detection engine
│   └── models.py                     ← Pydantic data models
├── frontend/
│   └── index.html                    ← Web UI
├── tests/
│   └── test_api.py                   ← Pytest suite
├── docs/
│   ├── API_REFERENCE.md
│   └── ARCHITECTURE.md
└── converge_widget_port8007.html     ← Standalone embeddable widget
```

---

**Author:** [Md Jisan Hossen](https://github.com/j1s4nn) — B.Sc. Artificial Intelligence, NUIST
**License:** MIT

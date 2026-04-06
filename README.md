# DuplicateIQ — Enterprise Duplicate Question Detection System

> Semantic duplicate detection powered by Sentence-BERT + FAISS. Understands *meaning*, not just matching words.

---

## 🏗 Architecture Overview

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
│  Frontend UI    │      │  Converge Widget        │
│  (Port 8000     │      │  converge_widget_       │
│   /frontend)    │      │  port8007.html          │
└─────────────────┘      └─────────────────────────┘
```

---

## 📁 Project Structure

```
duplicate-question-pairs/
├── 📋 README.md                          ← You are here
├── 📋 requirements.txt                   ← Python dependencies
├── 🐍 run.py                             ← Start the API server
│
├── backend/
│   ├── __init__.py
│   ├── main.py                           ← FastAPI routes + middleware
│   ├── engine.py                         ← SBERT + FAISS detection engine
│   └── models.py                         ← Pydantic data models
│
├── frontend/
│   └── index.html                        ← Full-featured web UI
│
├── tests/
│   ├── __init__.py
│   └── test_api.py                       ← Pytest test suite
│
├── docs/
│   ├── API_REFERENCE.md                  ← Full API documentation
│   └── ARCHITECTURE.md                   ← Deep-dive architecture notes
│
└── converge_widget_port8007.html         ← Standalone Converge tool
```

---

## 🚀 Quick Start

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

### 4. Add Converge widget
# this section for my website. though the website do not publish yet. but the website name is Converge.

Serve `converge_widget_port8007.html` on port 8007: 

```bash
python -m http.server 8007
# Then embed in Converge as an iframe or route
```

---

## 🔌 API Reference

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

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## ⚙️ Configuration

| Parameter | Default | Description |
|---|---|---|
| Model | `all-MiniLM-L6-v2` | SBERT model (balances speed + accuracy) |
| Threshold | `0.85` | Cosine similarity cutoff |
| Max batch | `50` | Pairs per batch request |
| Target latency | `<100ms` | Per-request inference target |

**Better accuracy** (slower): Use `all-mpnet-base-v2` in `engine.py`  
**Faster** (less accurate): Use `all-MiniLM-L12-v2`

---

## 📊 Key Design Decisions

**Why cosine similarity?**  
Captures directional similarity between embeddings — unaffected by embedding magnitude. Ideal for semantic matching.

**Why precision over recall?**  
False positives (merging unrelated questions) destroy user trust. We accept missing some duplicates to avoid wrongly merging distinct content.

**Why FAISS?**  
SQL `LIKE` queries can't do semantic search. FAISS enables sub-millisecond approximate nearest-neighbor search across millions of embeddings.

**Why Sentence-BERT?**  
Unlike plain BERT (which needs pairwise comparison), SBERT produces fixed-size sentence embeddings we can pre-compute and cache — enabling O(1) lookup per new question.

---

## 🛡 Fallback Mode

If `sentence-transformers` / FAISS are unavailable (e.g., CI/CD, lightweight environments), the engine automatically falls back to **TF-IDF + cosine similarity** (scikit-learn). All API contracts remain identical.

---

## 📦 Deployment

```bash
# Production with Gunicorn
pip install gunicorn
gunicorn backend.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

---

## 👤 Author

Developed by **Jisan** — *Full-Stack AI Developer*


> **Project Mission:** Built as an enterprise-grade NLP tool demonstrating:
> * 🧠 **Architecture:** Siamese neural networks
> * ⚡ **Search:** Semantic vector indexing at scale
> * ⚙️ **Backend:** Clean FastAPI design with async I/O
> * 🛡️ **Reliability:** Production-ready test coverage
> * 🎯 **Data Science:** Precision-first ML evaluation strategy

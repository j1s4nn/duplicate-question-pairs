# API Reference — DuplicateIQ

Base URL: `http://localhost:8000`  
Interactive docs: `http://localhost:8000/api/docs`

---

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check + engine status |
| POST | `/api/detect` | Compare a single question pair |
| POST | `/api/detect/batch` | Batch compare up to 50 pairs |
| POST | `/api/search` | Semantic similarity search against index |
| POST | `/api/index/question` | Add a question to the vector index |
| DELETE | `/api/index/reset` | Reset the index |
| GET | `/api/stats` | Usage statistics |

---

## Response Codes

| Code | Meaning |
|---|---|
| 200 | Success |
| 400 | Bad request (validation error) |
| 503 | Engine not ready (still loading) |

---

## Confidence Labels

| Label | Meaning |
|---|---|
| HIGH | Clearly duplicate / clearly not (far from threshold) |
| MEDIUM | Moderately confident |
| LOW | Close to the threshold boundary — consider manual review |

---

## Response Headers

Every response includes:
- `X-Request-ID` — Unique request identifier for tracing
- `X-Process-Time-Ms` — Server-side processing time in milliseconds

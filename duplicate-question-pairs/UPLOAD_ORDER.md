# 📋 Repository Upload Order — DuplicateIQ

Follow this order when pushing to GitHub for the cleanest commit history.

---

## STEP 1 — Project Foundation
Upload these first (repository setup):
- `README.md`
- `requirements.txt`
- `.gitignore` (if you create one — exclude __pycache__, .env, venv/)

---

## STEP 2 — Backend Core (Most impressive to HR)
Upload in this order:
1. `backend/__init__.py`
2. `backend/models.py`         ← Data contracts (clean Pydantic schemas)
3. `backend/engine.py`         ← Core ML logic (SBERT + FAISS) ⭐ STAR FILE
4. `backend/main.py`           ← FastAPI routes + middleware

---

## STEP 3 — Entry Point
- `run.py`

---

## STEP 4 — Tests
- `tests/__init__.py`
- `tests/test_api.py`          ← Full test coverage ⭐ HR loves this

---

## STEP 5 — Frontend
- `frontend/index.html`        ← Professional 5-tab UI ⭐

---

## STEP 6 — Converge Integration
- `converge_widget_port8007.html`

---

## STEP 7 — Documentation
- `docs/API_REFERENCE.md`
- `docs/ARCHITECTURE.md`

---

## 💡 Commit Message Suggestions

```
git init
git add README.md requirements.txt
git commit -m "chore: project scaffold and dependencies"

git add backend/
git commit -m "feat(backend): SBERT + FAISS duplicate detection engine with FastAPI"

git add tests/
git commit -m "test: full async test suite for engine and API endpoints"

git add frontend/
git commit -m "feat(frontend): 5-tab professional web UI for duplicate detection"

git add converge_widget_port8007.html
git commit -m "feat(widget): standalone Converge tool on port 8007"

git add docs/
git commit -m "docs: API reference and architecture deep-dive"
```

This gives a clean, professional commit history that shows:
✅ You think about project structure first
✅ Core logic is committed before UI
✅ Tests are written (not afterthoughts)
✅ Documentation exists

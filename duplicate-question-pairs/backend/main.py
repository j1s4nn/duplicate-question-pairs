"""
Duplicate Question Pairs Detection System
Enterprise-Grade FastAPI Backend
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import time
import logging
import uuid
from datetime import datetime

from .engine import DuplicateDetectionEngine
from .models import QuestionPair, DetectionResult, BatchRequest, BatchResult, HealthResponse

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("duplicate_detector")

# ─────────────────────────────────────────────
# App Init
# ─────────────────────────────────────────────
app = FastAPI(
    title="Duplicate Question Detection API",
    description="Enterprise-grade semantic duplicate detection using Sentence-BERT + FAISS",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine instance
engine: Optional[DuplicateDetectionEngine] = None


@app.on_event("startup")
async def startup_event():
    global engine
    logger.info("Initializing Duplicate Detection Engine...")
    engine = DuplicateDetectionEngine()
    await engine.initialize()
    logger.info("Engine ready ✓")


# ─────────────────────────────────────────────
# Middleware: Request ID + Timing
# ─────────────────────────────────────────────
@app.middleware("http")
async def add_request_metadata(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = str(duration_ms)
    logger.info(f"[{request_id}] {request.method} {request.url.path} → {response.status_code} ({duration_ms}ms)")
    return response


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """System health and engine status."""
    return HealthResponse(
        status="ok",
        engine_ready=engine is not None and engine.is_ready,
        model_name=engine.model_name if engine else "not loaded",
        questions_indexed=engine.question_count if engine else 0,
        timestamp=datetime.utcnow().isoformat()
    )


@app.post("/api/detect", response_model=DetectionResult, tags=["Detection"])
async def detect_duplicate(pair: QuestionPair):
    """
    Compare two questions and return similarity score + verdict.
    
    - **question_a**: First question string
    - **question_b**: Second question string  
    - **threshold**: Cosine similarity cutoff (default 0.85)
    """
    if not engine or not engine.is_ready:
        raise HTTPException(status_code=503, detail="Engine not ready. Please retry shortly.")
    
    start = time.perf_counter()
    result = await engine.compare(pair.question_a, pair.question_b, pair.threshold)
    result.latency_ms = round((time.perf_counter() - start) * 1000, 2)
    return result


@app.post("/api/detect/batch", response_model=BatchResult, tags=["Detection"])
async def detect_batch(batch: BatchRequest):
    """
    Compare multiple question pairs in one request. Max 50 pairs per batch.
    """
    if not engine or not engine.is_ready:
        raise HTTPException(status_code=503, detail="Engine not ready.")
    
    if len(batch.pairs) > 50:
        raise HTTPException(status_code=400, detail="Batch size cannot exceed 50 pairs.")
    
    start = time.perf_counter()
    results = []
    for pair in batch.pairs:
        r = await engine.compare(pair.question_a, pair.question_b, pair.threshold)
        results.append(r)
    
    total_ms = round((time.perf_counter() - start) * 1000, 2)
    duplicates = sum(1 for r in results if r.is_duplicate)
    
    return BatchResult(
        results=results,
        total_pairs=len(results),
        duplicates_found=duplicates,
        total_latency_ms=total_ms
    )


@app.post("/api/index/question", tags=["Index Management"])
async def index_question(payload: dict):
    """
    Add a new question to the vector index for future similarity search.
    """
    if not engine or not engine.is_ready:
        raise HTTPException(status_code=503, detail="Engine not ready.")
    
    question = payload.get("question", "").strip()
    question_id = payload.get("id", str(uuid.uuid4()))
    
    if not question:
        raise HTTPException(status_code=400, detail="Question text is required.")
    
    await engine.index_question(question_id, question)
    return {"status": "indexed", "id": question_id, "question": question}


@app.post("/api/search", tags=["Detection"])
async def search_similar(payload: dict):
    """
    Given a new question, find the top-K most similar questions already in the index.
    """
    if not engine or not engine.is_ready:
        raise HTTPException(status_code=503, detail="Engine not ready.")
    
    query = payload.get("question", "").strip()
    top_k = min(payload.get("top_k", 5), 20)
    threshold = payload.get("threshold", 0.75)
    
    if not query:
        raise HTTPException(status_code=400, detail="Question is required.")
    
    results = await engine.search_similar(query, top_k=top_k, threshold=threshold)
    return {
        "query": query,
        "matches": results,
        "total_indexed": engine.question_count
    }


@app.delete("/api/index/reset", tags=["Index Management"])
async def reset_index():
    """Reset the in-memory question index (demo use)."""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not ready.")
    await engine.reset_index()
    return {"status": "reset", "message": "Index cleared successfully."}


@app.get("/api/stats", tags=["System"])
async def get_stats():
    """Return usage statistics."""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not ready.")
    return engine.get_stats()

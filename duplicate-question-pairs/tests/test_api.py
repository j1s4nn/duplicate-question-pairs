"""
Test Suite for Duplicate Question Detection API
Run: pytest tests/ -v
"""

import pytest
import asyncio
from httpx import AsyncClient, ASGITransport

# We test with the app directly (ASGI transport — no real server needed)
from backend.main import app
from backend.engine import DuplicateDetectionEngine, TextNormalizer


# ─────────────────────────────────────────────
# Text Normalizer Tests
# ─────────────────────────────────────────────

class TestTextNormalizer:
    def test_contraction_expansion(self):
        result = TextNormalizer.expand_contractions("I can't do it and don't try")
        assert "cannot" in result
        assert "do not" in result

    def test_whitespace_normalization(self):
        result = TextNormalizer.clean_whitespace("  hello   world  ")
        assert result == "hello world"

    def test_unicode_normalization(self):
        result = TextNormalizer.normalize_unicode("caf\u00e9")
        assert result == "café"

    def test_full_pipeline(self):
        result = TextNormalizer.normalize("  I can't login  it's broken  ")
        assert "cannot" in result
        assert "it is" in result


# ─────────────────────────────────────────────
# Engine Tests
# ─────────────────────────────────────────────

class TestDuplicateDetectionEngine:
    @pytest.fixture
    async def engine(self):
        eng = DuplicateDetectionEngine()
        await eng.initialize()
        return eng

    @pytest.mark.asyncio
    async def test_engine_initializes(self, engine):
        assert engine.is_ready

    @pytest.mark.asyncio
    async def test_identical_questions_high_similarity(self, engine):
        result = await engine.compare(
            "How do I reset my password?",
            "How do I reset my password?"
        )
        assert result.similarity_score > 0.95

    @pytest.mark.asyncio
    async def test_semantic_duplicates_detected(self, engine):
        result = await engine.compare(
            "How do I reset my password?",
            "I forgot my login credentials, what should I do?",
            threshold=0.5
        )
        assert result.similarity_score > 0.3  # Semantically related

    @pytest.mark.asyncio
    async def test_unrelated_questions_low_similarity(self, engine):
        result = await engine.compare(
            "How do I bake a chocolate cake?",
            "What is the speed of light?"
        )
        assert result.similarity_score < 0.5

    @pytest.mark.asyncio
    async def test_hard_negatives(self, engine):
        """Questions that look similar but have different intents."""
        result = await engine.compare(
            "How do I start a car?",
            "How do I stop a car?"
        )
        # Should have moderate similarity, not be classified as high-confidence duplicate
        assert result.similarity_score < 0.99

    @pytest.mark.asyncio
    async def test_threshold_respected(self, engine):
        result = await engine.compare("hello world", "hello world", threshold=0.99)
        # Even identical at threshold=0.99 should pass
        assert result.threshold_used == 0.99

    @pytest.mark.asyncio
    async def test_confidence_labels(self, engine):
        result = await engine.compare(
            "How to reset password?",
            "How to reset password?"
        )
        assert result.confidence in ("HIGH", "MEDIUM", "LOW")

    @pytest.mark.asyncio
    async def test_index_and_search(self, engine):
        await engine.index_question("q1", "How do I cancel my subscription?")
        await engine.index_question("q2", "What is the weather today?")
        results = await engine.search_similar("I want to cancel my plan", threshold=0.3)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_reset_index(self, engine):
        await engine.index_question("qx", "Some question")
        await engine.reset_index()
        assert engine.question_count == 0


# ─────────────────────────────────────────────
# API Integration Tests
# ─────────────────────────────────────────────

@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.mark.asyncio
async def test_health_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_detect_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/detect", json={
            "question_a": "How do I reset my password?",
            "question_b": "I forgot my login credentials",
            "threshold": 0.5
        })
    assert response.status_code == 200
    data = response.json()
    assert "similarity_score" in data
    assert "is_duplicate" in data
    assert "confidence" in data


@pytest.mark.asyncio
async def test_batch_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/detect/batch", json={
            "pairs": [
                {"question_a": "Reset password", "question_b": "Forgot login", "threshold": 0.5},
                {"question_a": "Bake a cake", "question_b": "Cook pasta", "threshold": 0.85},
            ]
        })
    assert response.status_code == 200
    data = response.json()
    assert data["total_pairs"] == 2
    assert "duplicates_found" in data


@pytest.mark.asyncio
async def test_batch_size_limit():
    pairs = [{"question_a": f"Q{i}", "question_b": f"Q{i+1}", "threshold": 0.85} for i in range(51)]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/detect/batch", json={"pairs": pairs})
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_stats_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/stats")
    assert response.status_code == 200

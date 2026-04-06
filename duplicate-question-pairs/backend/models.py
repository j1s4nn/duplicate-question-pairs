"""
Pydantic Data Models for Duplicate Question Detection API
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class QuestionPair(BaseModel):
    question_a: str = Field(..., min_length=3, max_length=1000, example="How do I reset my password?")
    question_b: str = Field(..., min_length=3, max_length=1000, example="I forgot my login credentials, what should I do?")
    threshold: float = Field(default=0.85, ge=0.0, le=1.0, description="Cosine similarity cutoff for duplicate verdict")

    class Config:
        schema_extra = {
            "example": {
                "question_a": "How do I reset my password?",
                "question_b": "I forgot my login credentials, what should I do?",
                "threshold": 0.85
            }
        }


class SimilarQuestion(BaseModel):
    id: str
    question: str
    similarity_score: float
    is_duplicate: bool


class DetectionResult(BaseModel):
    question_a: str
    question_b: str
    similarity_score: float = Field(..., description="Cosine similarity score between 0 and 1")
    is_duplicate: bool = Field(..., description="True if similarity_score >= threshold")
    threshold_used: float
    confidence: str = Field(..., description="HIGH / MEDIUM / LOW confidence label")
    latency_ms: Optional[float] = None

    class Config:
        schema_extra = {
            "example": {
                "question_a": "How do I reset my password?",
                "question_b": "I forgot my login credentials, what should I do?",
                "similarity_score": 0.91,
                "is_duplicate": True,
                "threshold_used": 0.85,
                "confidence": "HIGH",
                "latency_ms": 38.4
            }
        }


class BatchRequest(BaseModel):
    pairs: List[QuestionPair] = Field(..., max_items=50)


class BatchResult(BaseModel):
    results: List[DetectionResult]
    total_pairs: int
    duplicates_found: int
    total_latency_ms: float


class HealthResponse(BaseModel):
    status: str
    engine_ready: bool
    model_name: str
    questions_indexed: int
    timestamp: str

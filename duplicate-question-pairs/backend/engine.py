"""
Core Duplicate Detection Engine
Implements Siamese Neural Network architecture using Sentence-BERT + FAISS
"""

import asyncio
import logging
import time
import re
import unicodedata
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import numpy as np

logger = logging.getLogger("duplicate_detector.engine")

# ─────────────────────────────────────────────
# Contraction Map for Normalization
# ─────────────────────────────────────────────
CONTRACTIONS = {
    "can't": "cannot", "won't": "will not", "don't": "do not",
    "doesn't": "does not", "didn't": "did not", "isn't": "is not",
    "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
    "wouldn't": "would not", "shouldn't": "should not", "couldn't": "could not",
    "mustn't": "must not", "i'm": "i am", "i've": "i have", "i'll": "i will",
    "i'd": "i would", "you're": "you are", "you've": "you have",
    "you'll": "you will", "they're": "they are", "we're": "we are",
    "he's": "he is", "she's": "she is", "it's": "it is", "that's": "that is",
    "there's": "there is", "here's": "here is", "what's": "what is",
    "let's": "let us", "who's": "who is", "how's": "how is",
}


class TextNormalizer:
    """
    Handles text preprocessing: contraction expansion, 
    unicode normalization, domain jargon handling.
    """

    @staticmethod
    def expand_contractions(text: str) -> str:
        pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in CONTRACTIONS) + r')\b', re.IGNORECASE)
        def replace(match):
            word = match.group(0).lower()
            return CONTRACTIONS.get(word, word)
        return pattern.sub(replace, text)

    @staticmethod
    def normalize_unicode(text: str) -> str:
        return unicodedata.normalize("NFKC", text)

    @staticmethod
    def clean_whitespace(text: str) -> str:
        return re.sub(r'\s+', ' ', text).strip()

    @staticmethod
    def normalize(text: str) -> str:
        text = TextNormalizer.normalize_unicode(text)
        text = TextNormalizer.expand_contractions(text)
        text = TextNormalizer.clean_whitespace(text)
        return text


class DuplicateDetectionEngine:
    """
    Enterprise-grade duplicate detection engine.
    
    Architecture:
      - Sentence-BERT (SBERT) for semantic embeddings
      - FAISS for approximate nearest-neighbor search at scale
      - Cosine similarity for pairwise comparison
      - Threshold tuning with confidence labeling
    
    Fallback:
      - When torch/sentence-transformers are unavailable (demo/test env),
        falls back to TF-IDF + cosine similarity for functional demonstration.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", threshold: float = 0.85):
        self.model_name = model_name
        self.default_threshold = threshold
        self.is_ready = False
        self._model = None
        self._use_fallback = False
        self._index = None           # FAISS index
        self._id_map: Dict[int, str] = {}        # faiss_idx → question_id
        self._question_map: Dict[str, str] = {}  # question_id → question text
        self._embeddings_cache: Dict[str, np.ndarray] = {}
        self._stats = defaultdict(int)
        self._normalizer = TextNormalizer()

    # ─────────────────────────────────────────────
    # Init
    # ─────────────────────────────────────────────

    async def initialize(self):
        """Load model and initialize FAISS index."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model)
        self.is_ready = True
        logger.info(f"Engine initialized | Model: {self.model_name} | Fallback: {self._use_fallback}")

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
            self._model = SentenceTransformer(self.model_name)
            dim = self._model.get_sentence_embedding_dimension()
            self._index = faiss.IndexFlatIP(dim)  # Inner Product = cosine after normalization
            self._use_fallback = False
            logger.info(f"SBERT model loaded: {self.model_name} | Embedding dim: {dim}")
        except ImportError:
            logger.warning("sentence-transformers or FAISS not available — using TF-IDF fallback")
            self._load_tfidf_fallback()

    def _load_tfidf_fallback(self):
        """TF-IDF + cosine similarity fallback for environments without GPU/torch."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        self._tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self._cosine_fn = cosine_similarity
        self._fallback_corpus: List[str] = []
        self._fallback_ids: List[str] = []
        self._use_fallback = True
        self.model_name = "TF-IDF (fallback)"

    # ─────────────────────────────────────────────
    # Embedding
    # ─────────────────────────────────────────────

    def _embed(self, text: str) -> np.ndarray:
        """Return L2-normalized embedding for a single text."""
        if text in self._embeddings_cache:
            return self._embeddings_cache[text]

        normalized_text = self._normalizer.normalize(text)

        if self._use_fallback:
            vec = self._tfidf_embed(normalized_text)
        else:
            vec = self._model.encode([normalized_text], convert_to_numpy=True, normalize_embeddings=True)[0]

        self._embeddings_cache[text] = vec
        return vec

    def _tfidf_embed(self, text: str) -> np.ndarray:
        corpus = self._fallback_corpus + [text]
        matrix = self._tfidf.fit_transform(corpus)
        vec = matrix[-1].toarray()[0]
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    # ─────────────────────────────────────────────
    # Core Compare
    # ─────────────────────────────────────────────

    async def compare(self, question_a: str, question_b: str, threshold: Optional[float] = None) -> "DetectionResult":
        from .models import DetectionResult
        threshold = threshold if threshold is not None else self.default_threshold

        loop = asyncio.get_event_loop()
        score = await loop.run_in_executor(None, self._cosine_similarity, question_a, question_b)
        score = round(float(score), 4)

        is_duplicate = score >= threshold
        confidence = self._confidence_label(score, threshold)

        self._stats["total_comparisons"] += 1
        if is_duplicate:
            self._stats["duplicates_detected"] += 1

        return DetectionResult(
            question_a=question_a,
            question_b=question_b,
            similarity_score=score,
            is_duplicate=is_duplicate,
            threshold_used=threshold,
            confidence=confidence,
        )

    def _cosine_similarity(self, text_a: str, text_b: str) -> float:
        if self._use_fallback:
            norm_a = self._normalizer.normalize(text_a)
            norm_b = self._normalizer.normalize(text_b)
            matrix = self._tfidf.fit_transform([norm_a, norm_b])
            arr = matrix.toarray()
            dot = np.dot(arr[0], arr[1])
            norms = np.linalg.norm(arr[0]) * np.linalg.norm(arr[1])
            return float(dot / norms) if norms > 0 else 0.0
        else:
            emb_a = self._embed(text_a)
            emb_b = self._embed(text_b)
            return float(np.dot(emb_a, emb_b))  # Already normalized → inner product = cosine

    @staticmethod
    def _confidence_label(score: float, threshold: float) -> str:
        gap = score - threshold
        if score >= threshold:
            if gap >= 0.10:
                return "HIGH"
            elif gap >= 0.04:
                return "MEDIUM"
            else:
                return "LOW"
        else:
            if abs(gap) <= 0.05:
                return "LOW"   # borderline non-duplicate
            return "HIGH"      # confidently not duplicate

    # ─────────────────────────────────────────────
    # Index Management
    # ─────────────────────────────────────────────

    async def index_question(self, question_id: str, question: str):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._add_to_index, question_id, question)

    def _add_to_index(self, question_id: str, question: str):
        if self._use_fallback:
            self._fallback_corpus.append(self._normalizer.normalize(question))
            self._fallback_ids.append(question_id)
        else:
            vec = self._embed(question)
            vec = vec.reshape(1, -1).astype(np.float32)
            faiss_idx = self._index.ntotal
            self._index.add(vec)
            self._id_map[faiss_idx] = question_id
        self._question_map[question_id] = question
        self._stats["questions_indexed"] += 1

    async def search_similar(self, query: str, top_k: int = 5, threshold: float = 0.75) -> List[dict]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._search, query, top_k, threshold)

    def _search(self, query: str, top_k: int, threshold: float) -> List[dict]:
        if not self._question_map:
            return []

        results = []
        if self._use_fallback:
            for qid, qtext in self._question_map.items():
                score = self._cosine_similarity(query, qtext)
                if score >= threshold:
                    results.append({
                        "id": qid,
                        "question": qtext,
                        "similarity_score": round(score, 4),
                        "is_duplicate": True
                    })
            results.sort(key=lambda x: x["similarity_score"], reverse=True)
            return results[:top_k]
        else:
            vec = self._embed(query).reshape(1, -1).astype(np.float32)
            k = min(top_k, self._index.ntotal)
            if k == 0:
                return []
            scores, indices = self._index.search(vec, k)
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                qid = self._id_map.get(int(idx))
                if qid and score >= threshold:
                    results.append({
                        "id": qid,
                        "question": self._question_map[qid],
                        "similarity_score": round(float(score), 4),
                        "is_duplicate": True
                    })
            return results

    async def reset_index(self):
        if not self._use_fallback and self._index is not None:
            import faiss
            dim = self._index.d
            self._index = faiss.IndexFlatIP(dim)
        self._id_map.clear()
        self._question_map.clear()
        self._fallback_corpus = []
        self._fallback_ids = []
        self._embeddings_cache.clear()
        self._stats["questions_indexed"] = 0
        logger.info("Index reset.")

    # ─────────────────────────────────────────────
    # Stats
    # ─────────────────────────────────────────────

    @property
    def question_count(self) -> int:
        return len(self._question_map)

    def get_stats(self) -> dict:
        return {
            "model": self.model_name,
            "fallback_mode": self._use_fallback,
            "questions_indexed": self.question_count,
            "total_comparisons": self._stats["total_comparisons"],
            "duplicates_detected": self._stats["duplicates_detected"],
            "cache_size": len(self._embeddings_cache),
        }

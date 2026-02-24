"""
Embeddings — Sentence Transformer embeddings for RAG.
Uses all-MiniLM-L6-v2 for fast, high-quality embeddings.
Provides a ChromaDB-compatible embedding function.
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

_embedding_model = None


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """Load or return cached sentence transformer model."""
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer(model_name)
            logger.info(f"Embedding model loaded: {model_name}")
        except ImportError:
            logger.error("sentence-transformers not installed")
            return None
    return _embedding_model


def generate_embeddings(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> Optional[List[List[float]]]:
    """Generate embeddings for a list of texts."""
    model = get_embedding_model(model_name)
    if model is None:
        return None
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return embeddings.tolist()


def generate_single_embedding(text: str, model_name: str = "all-MiniLM-L6-v2") -> Optional[List[float]]:
    """Generate embedding for a single text."""
    result = generate_embeddings([text], model_name)
    return result[0] if result else None


class SentenceTransformerEmbeddingFunction:
    """
    ChromaDB-compatible embedding function backed by sentence-transformers.
    Implements the __call__ protocol that ChromaDB expects.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name

    def __call__(self, input: List[str]) -> List[List[float]]:
        """Generate embeddings for ChromaDB. Accepts a list of documents/queries."""
        result = generate_embeddings(input, self._model_name)
        if result is None:
            # Return zero vectors as fallback so ChromaDB doesn't crash
            logger.warning("Embedding generation failed, returning zero vectors")
            return [[0.0] * 384 for _ in input]  # MiniLM-L6-v2 dim = 384
        return result

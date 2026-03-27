"""
Local sentence-transformers embedding model for semantic search.
Lazy-loads the model on first use to avoid startup overhead.

Default model: all-MiniLM-L6-v2 (22 MB, 384-dim, ~14 ms/sentence on CPU)
Override with EMBED_MODEL_NAME env var.
"""
from __future__ import annotations

import struct
import threading

from ..config import Config
from ..utils.logger import get_logger

logger = get_logger("mirofish.local_zep.embedder")

_model = None
_model_lock = threading.Lock()

EMBEDDING_DIM = 384
_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer
                model_name = getattr(Config, "EMBED_MODEL_NAME", _DEFAULT_MODEL)
                logger.info(f"Loading embedding model: {model_name}")
                _model = SentenceTransformer(model_name)
                logger.info("Embedding model ready")
    return _model


def encode(text: str) -> bytes:
    """Encode text to a float32 bytes blob suitable for sqlite-vec storage.

    Returns empty bytes on failure (caller should treat as no embedding).
    """
    if not text or not text.strip():
        return b""
    try:
        model = _get_model()
        vec = model.encode(text.strip(), normalize_embeddings=True)
        return struct.pack(f"{len(vec)}f", *vec)
    except Exception as e:
        logger.warning(f"Embedding encode failed: {e}")
        return b""

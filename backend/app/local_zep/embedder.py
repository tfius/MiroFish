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

try:
    from sentence_transformers import SentenceTransformer
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False

_model = None
_model_lock = threading.Lock()

EMBEDDING_DIM = 384
_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                model_name = getattr(Config, "EMBED_MODEL_NAME", _DEFAULT_MODEL)
                logger.info(f"Loading embedding model: {model_name}")
                m = SentenceTransformer(model_name)
                # Validate output dimension matches expected EMBEDDING_DIM
                test_vec = m.encode("test", normalize_embeddings=True)
                if len(test_vec) != EMBEDDING_DIM:
                    raise ValueError(
                        f"Embedding model '{model_name}' produces {len(test_vec)}-dim vectors "
                        f"but EMBEDDING_DIM={EMBEDDING_DIM}. "
                        f"Set EMBED_MODEL_NAME to a {EMBEDDING_DIM}-dim model or update EMBEDDING_DIM."
                    )
                _model = m
                logger.info(f"Embedding model ready ({EMBEDDING_DIM}-dim)")
    return _model


def encode(text: str) -> bytes:
    """Encode text to a float32 bytes blob suitable for sqlite-vec storage.

    Returns empty bytes if sentence-transformers is not installed, text is empty,
    or inference fails. Callers must treat empty bytes as 'no embedding'.
    """
    if not _SENTENCE_TRANSFORMERS_AVAILABLE:
        return b""
    if not text or not text.strip():
        return b""
    try:
        model = _get_model()
        vec = model.encode(text.strip(), normalize_embeddings=True)
        return struct.pack(f"{len(vec)}f", *vec)
    except Exception as e:
        logger.warning(f"Embedding encode failed: {e}")
        return b""

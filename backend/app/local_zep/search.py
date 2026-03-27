"""
local_zep hybrid search: FTS5 trigram + sentence-transformers vector search.

Strategy: always run both passes, merge with Reciprocal Rank Fusion (RRF).
RRF score = Σ 1/(k + rank), k=60 (standard default).
FTS results are interleaved with vector results by score rather than appended,
so the top results reflect both keyword precision and semantic relevance.
"""
from __future__ import annotations

from ..utils.logger import get_logger
from . import db
from . import embedder
from .models import SearchResponse

logger = get_logger("mirofish.local_zep.search")

_RRF_K = 60  # standard RRF constant


def _rrf_merge(lists: list, limit: int) -> list:
    """Reciprocal Rank Fusion across multiple ranked result lists.

    Each list contributes 1/(k + rank+1) to its items' scores.
    Items appearing in multiple lists accumulate scores from all lists.
    """
    scores: dict = {}
    items: dict = {}
    for lst in lists:
        for rank, item in enumerate(lst):
            uid = item.uuid_
            scores[uid] = scores.get(uid, 0.0) + 1.0 / (_RRF_K + rank + 1)
            if uid not in items:
                items[uid] = item
    ranked = sorted(scores, key=lambda uid: scores[uid], reverse=True)
    return [items[uid] for uid in ranked[:limit]]


def search(
    graph_ids: list,
    query: str,
    limit: int = 20,
    scope: str = "edges",
) -> SearchResponse:
    """
    Hybrid RRF search: FTS5 trigram + vector cosine similarity.
    scope='edges' → populates SearchResponse.edges
    scope='nodes' → populates SearchResponse.nodes
    """
    if not graph_ids or not query:
        return SearchResponse()

    # Encode query once for vector pass (empty bytes if embedder unavailable)
    query_emb = embedder.encode(query) if db._SQLITE_VEC_AVAILABLE else b""

    if scope == "nodes":
        fts_results = []
        try:
            fts_results = db.search_nodes_fts(graph_ids, query, limit)
        except Exception as e:
            logger.warning(f"FTS node search failed: {e}")

        if not fts_results:
            try:
                fts_results = db.search_nodes_keyword(graph_ids, query, limit)
            except Exception as e:
                logger.warning(f"Keyword node search failed: {e}")

        vec_results = db.search_nodes_vec(graph_ids, query_emb, limit) if query_emb else []

        results = _rrf_merge([fts_results, vec_results], limit) if vec_results else fts_results[:limit]
        return SearchResponse(nodes=results)

    else:
        fts_results = []
        try:
            fts_results = db.search_edges_fts(graph_ids, query, limit)
        except Exception as e:
            logger.warning(f"FTS edge search failed: {e}")

        if not fts_results:
            try:
                fts_results = db.search_edges_keyword(graph_ids, query, limit)
            except Exception as e:
                logger.warning(f"Keyword edge search failed: {e}")

        vec_results = db.search_edges_vec(graph_ids, query_emb, limit) if query_emb else []

        results = _rrf_merge([fts_results, vec_results], limit) if vec_results else fts_results[:limit]
        return SearchResponse(edges=results)

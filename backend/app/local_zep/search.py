"""
local_zep hybrid search: FTS5 trigram + sentence-transformers vector search.

Strategy:
  1. FTS5 (keyword match) — fast, exact substring/token matching
  2. Vector (cosine similarity) — semantic matching via local embeddings,
     only runs when FTS returned fewer results than the requested limit
  3. Merge: FTS5 results first (higher precision), then vector-only hits to fill up to limit
"""
from __future__ import annotations

from ..utils.logger import get_logger
from . import db
from . import embedder
from .models import SearchResponse

logger = get_logger("mirofish.local_zep.search")


def _merge_unique(primary: list, secondary: list, limit: int, key: str = "uuid_") -> list:
    """FTS results first, then vector-only results to fill remaining slots up to limit.

    If primary already has `limit` items, secondary is not added (it would all be cut anyway).
    """
    if len(primary) >= limit:
        return primary[:limit]
    seen = {getattr(r, key) for r in primary}
    extra = [r for r in secondary if getattr(r, key) not in seen]
    return (primary + extra)[:limit]


def search(
    graph_ids: list,
    query: str,
    limit: int = 20,
    scope: str = "edges",
) -> SearchResponse:
    """
    Hybrid search: FTS5 trigram + vector cosine similarity.
    scope='edges' → populates SearchResponse.edges
    scope='nodes' → populates SearchResponse.nodes
    """
    if not graph_ids or not query:
        return SearchResponse()

    if scope == "nodes":
        # FTS5 pass
        fts_results = []
        try:
            fts_results = db.search_nodes_fts(graph_ids, query, limit)
        except Exception as e:
            logger.warning(f"FTS node search failed: {e}")

        # Keyword fallback if FTS empty
        if not fts_results:
            try:
                fts_results = db.search_nodes_keyword(graph_ids, query, limit)
            except Exception as e:
                logger.warning(f"Keyword node search failed: {e}")

        # Vector pass — only if FTS didn't fill the limit and sqlite-vec is available
        vec_results = []
        if len(fts_results) < limit and db._SQLITE_VEC_AVAILABLE:
            query_emb = embedder.encode(query)
            vec_results = db.search_nodes_vec(graph_ids, query_emb, limit)

        return SearchResponse(nodes=_merge_unique(fts_results, vec_results, limit))

    else:
        # FTS5 pass
        fts_results = []
        try:
            fts_results = db.search_edges_fts(graph_ids, query, limit)
        except Exception as e:
            logger.warning(f"FTS edge search failed: {e}")

        # Keyword fallback if FTS empty
        if not fts_results:
            try:
                fts_results = db.search_edges_keyword(graph_ids, query, limit)
            except Exception as e:
                logger.warning(f"Keyword edge search failed: {e}")

        # Vector pass — only if FTS didn't fill the limit and sqlite-vec is available
        vec_results = []
        if len(fts_results) < limit and db._SQLITE_VEC_AVAILABLE:
            query_emb = embedder.encode(query)
            vec_results = db.search_edges_vec(graph_ids, query_emb, limit)

        return SearchResponse(edges=_merge_unique(fts_results, vec_results, limit))

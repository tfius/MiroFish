"""
local_zep hybrid search: FTS5 trigram + sentence-transformers vector search.

Strategy:
  1. FTS5 (keyword match) — fast, exact substring/token matching
  2. Vector (cosine similarity) — semantic matching via local embeddings
  3. Merge: FTS5 results first (higher precision), then vector-only hits up to limit
"""
from __future__ import annotations

from ..utils.logger import get_logger
from . import db
from . import embedder
from .models import SearchResponse

logger = get_logger("mirofish.local_zep.search")


def _merge_unique(primary: list, secondary: list, limit: int, key: str = "uuid_") -> list:
    """Return primary results followed by secondary-only results, deduplicated, up to limit."""
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

    # Encode query once for vector search
    query_emb = embedder.encode(query)

    if scope == "nodes":
        # FTS5 pass
        try:
            fts_results = db.search_nodes_fts(graph_ids, query, limit)
        except Exception as e:
            logger.warning(f"FTS node search failed: {e}")
            fts_results = []

        # Keyword fallback if FTS empty
        if not fts_results:
            fts_results = db.search_nodes_keyword(graph_ids, query, limit)

        # Vector pass
        vec_results = db.search_nodes_vec(graph_ids, query_emb, limit)

        results = _merge_unique(fts_results, vec_results, limit)
        return SearchResponse(nodes=results)

    else:
        # FTS5 pass
        try:
            fts_results = db.search_edges_fts(graph_ids, query, limit)
        except Exception as e:
            logger.warning(f"FTS edge search failed: {e}")
            fts_results = []

        # Keyword fallback if FTS empty
        if not fts_results:
            fts_results = db.search_edges_keyword(graph_ids, query, limit)

        # Vector pass
        vec_results = db.search_edges_vec(graph_ids, query_emb, limit)

        results = _merge_unique(fts_results, vec_results, limit)
        return SearchResponse(edges=results)

"""
local_zep FTS5 搜索
"""

from __future__ import annotations

from ..utils.logger import get_logger
from . import db
from .models import SearchResponse

logger = get_logger("mirofish.local_zep.search")


def search(
    graph_ids: list,
    query: str,
    limit: int = 20,
    scope: str = "edges",
) -> SearchResponse:
    """
    scope='edges' → SearchResponse.edges 填充
    scope='nodes' → SearchResponse.nodes 填充
    """
    if not graph_ids or not query:
        return SearchResponse()

    if scope == "nodes":
        try:
            results = db.search_nodes_fts(graph_ids, query, limit)
        except Exception as e:
            logger.warning(f"FTS node search failed, falling back to keyword search: {e}")
            results = []
        if not results:
            results = db.search_nodes_keyword(graph_ids, query, limit)
        return SearchResponse(nodes=results)
    else:
        try:
            results = db.search_edges_fts(graph_ids, query, limit)
        except Exception as e:
            logger.warning(f"FTS edge search failed, falling back to keyword search: {e}")
            results = []
        if not results:
            results = db.search_edges_keyword(graph_ids, query, limit)
        return SearchResponse(edges=results)

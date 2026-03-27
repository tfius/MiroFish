"""
local_zep data models
Dataclasses compatible with the zep_cloud SDK interface.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class NodeResponse:
    uuid_: str
    graph_id: str
    name: str
    labels: list
    summary: str
    attributes: dict
    created_at: Optional[str] = None


@dataclass
class EdgeResponse:
    uuid_: str
    graph_id: str
    name: str
    fact: str
    fact_type: str
    source_node_uuid: str
    target_node_uuid: str
    attributes: dict
    valid_at: Optional[str] = None
    invalid_at: Optional[str] = None
    expired_at: Optional[str] = None
    created_at: Optional[str] = None
    episodes: Optional[list] = None


@dataclass
class EpisodeResponse:
    uuid_: str
    graph_id: str
    data: str
    type: str
    processed: bool
    created_at: Optional[str] = None


class EpisodeData:
    """Matches the zep_cloud.EpisodeData interface."""
    def __init__(self, data: str, type: str = "text"):
        self.data = data
        self.type = type


class EntityEdgeSourceTarget:
    """Matches the zep_cloud.EntityEdgeSourceTarget interface."""
    def __init__(self, source: str, target: str):
        self.source = source
        self.target = target


class InternalServerError(Exception):
    """Matches zep_cloud.InternalServerError; used by zep_paging.py retry logic."""
    pass


@dataclass
class SearchResponse:
    """Result object returned by graph.search()."""
    edges: list = field(default_factory=list)
    nodes: list = field(default_factory=list)

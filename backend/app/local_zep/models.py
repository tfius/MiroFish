"""
local_zep 数据模型
与 zep_cloud SDK 接口兼容的数据类
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class NodeResponse:
    uuid_: str
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
    """匹配 zep_cloud.EpisodeData 接口"""
    def __init__(self, data: str, type: str = "text"):
        self.data = data
        self.type = type


class EntityEdgeSourceTarget:
    """匹配 zep_cloud.EntityEdgeSourceTarget 接口"""
    def __init__(self, source: str, target: str):
        self.source = source
        self.target = target


class InternalServerError(Exception):
    """匹配 zep_cloud.InternalServerError，用于 zep_paging.py 重试逻辑"""
    pass


@dataclass
class SearchResponse:
    """graph.search() 返回的结果对象"""
    edges: list = field(default_factory=list)
    nodes: list = field(default_factory=list)

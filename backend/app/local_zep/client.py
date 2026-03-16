"""
local_zep 客户端
LocalZep 类精确匹配 zep_cloud.client.Zep 的接口
"""

from __future__ import annotations

import uuid as _uuid
from typing import Any, Optional

from . import db, extractor, search as _search
from .models import EpisodeData, EpisodeResponse, SearchResponse


class _EpisodeNamespace:
    def get(self, uuid_: str) -> Optional[EpisodeResponse]:
        return db.get_episode(uuid_)


class _NodeNamespace:
    def get(self, uuid_: str):
        return db.get_node(uuid_)

    def get_by_graph_id(self, graph_id: str, limit: int = 100, uuid_cursor: Optional[str] = None):
        return db.get_nodes_page(graph_id, limit, uuid_cursor)

    def get_entity_edges(self, node_uuid: str):
        return db.get_node_edges(node_uuid)


class _EdgeNamespace:
    def get_by_graph_id(self, graph_id: str, limit: int = 100, uuid_cursor: Optional[str] = None):
        return db.get_edges_page(graph_id, limit, uuid_cursor)


class _GraphNamespace:
    def __init__(self):
        self.node = _NodeNamespace()
        self.edge = _EdgeNamespace()
        self.episode = _EpisodeNamespace()

    # ── graph CRUD ──────────────────────────────────────────────────────────

    def create(self, graph_id: str, name: str = "", description: str = ""):
        db.create_graph(graph_id, name, description)

    def delete(self, graph_id: str):
        db.delete_graph(graph_id)

    # ── episode add ─────────────────────────────────────────────────────────

    def add(self, graph_id: str, type: str = "text", data: str = "") -> EpisodeResponse:
        ep_uuid = str(_uuid.uuid4())
        db.create_episode(ep_uuid, graph_id, data, type)
        extractor.submit_extraction(ep_uuid, graph_id, data)
        return db.get_episode(ep_uuid)

    def add_batch(self, graph_id: str, episodes: list) -> list[EpisodeResponse]:
        results = []
        for ep in episodes:
            data = ep.data if hasattr(ep, "data") else str(ep)
            type_ = ep.type if hasattr(ep, "type") else "text"
            ep_uuid = str(_uuid.uuid4())
            db.create_episode(ep_uuid, graph_id, data, type_)
            extractor.submit_extraction(ep_uuid, graph_id, data)
            results.append(db.get_episode(ep_uuid))
        return results

    # ── ontology ────────────────────────────────────────────────────────────

    def set_ontology(
        self,
        graph_ids: list,
        entities: Optional[dict] = None,
        edges: Optional[dict] = None,
    ):
        """
        从动态类中提取本体信息并存储到数据库。
        entities: dict[str, type]  (EntityModel 子类)
        edges:    dict[str, tuple(type, list[EntityEdgeSourceTarget])]
        """
        ontology: dict[str, Any] = {"entity_types": [], "edge_types": []}

        if entities:
            for name, cls in entities.items():
                doc = getattr(cls, "__doc__", "") or ""
                annotations = getattr(cls, "__annotations__", {})
                attrs = []
                for attr_name, _ in annotations.items():
                    field_obj = cls.__dict__.get(attr_name)
                    desc = ""
                    if hasattr(field_obj, "description"):
                        desc = field_obj.description or attr_name
                    attrs.append({"name": attr_name, "description": desc})
                ontology["entity_types"].append({
                    "name": name,
                    "description": doc,
                    "attributes": attrs,
                })

        if edges:
            for name, value in edges.items():
                if isinstance(value, tuple):
                    cls, source_targets = value[0], value[1]
                else:
                    cls, source_targets = value, []
                doc = getattr(cls, "__doc__", "") or ""
                sts = []
                for st in source_targets:
                    sts.append({"source": st.source, "target": st.target})
                ontology["edge_types"].append({
                    "name": name,
                    "description": doc,
                    "source_targets": sts,
                })

        for gid in graph_ids:
            db.set_graph_ontology(gid, ontology)

    # ── search ──────────────────────────────────────────────────────────────

    def search(
        self,
        graph_ids: Optional[list] = None,
        graph_id: Optional[str] = None,
        text: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 20,
        scope: str = "edges",
        reranker: Optional[str] = None,
    ) -> SearchResponse:
        gids = graph_ids or ([graph_id] if graph_id else [])
        q = text or query or ""
        return _search.search(gids, q, limit, scope)


class LocalZep:
    """
    本地 Zep 替代品，接口与 zep_cloud.client.Zep 完全兼容。
    api_key 参数被接受但忽略（用于无缝替换）。
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.graph = _GraphNamespace()

"""
local_zep SQLite 存储层
WAL 模式 + FTS5 trigram 索引，线程安全写操作

改进：
- upsert_node 合并 labels/summary/attributes（不覆盖）
- create_edge 设置 valid_at，并将相同关系的旧边标记为 expired/invalid
  → 启用 PanoramaSearch 的历史事实区分功能
- search_*_keyword 关键词兜底搜索（FTS5 无结果时使用）
- get_graph_stats 用 COUNT 高效统计
"""

import json
import os
import re as _re
import sqlite3
import threading
import uuid as _uuid
from datetime import datetime, timezone
from typing import Any, Optional

from .models import EdgeResponse, EpisodeResponse, NodeResponse

_DB_PATH: Optional[str] = None
_lock = threading.Lock()
_local = threading.local()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init(db_path: str):
    global _DB_PATH
    _DB_PATH = db_path
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    with _lock:
        conn = _get_conn()
        _create_schema(conn)


def _get_conn() -> sqlite3.Connection:
    """每个线程维护自己的连接（WAL 允许多读一写）"""
    if not hasattr(_local, "conn") or _local.conn is None:
        conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        _local.conn = conn
    return _local.conn


def _create_schema(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS graphs (
            graph_id   TEXT PRIMARY KEY,
            name       TEXT,
            description TEXT,
            ontology   TEXT,
            created_at TEXT
        );

        CREATE TABLE IF NOT EXISTS nodes (
            uuid       TEXT PRIMARY KEY,
            graph_id   TEXT NOT NULL REFERENCES graphs(graph_id) ON DELETE CASCADE,
            name       TEXT NOT NULL,
            name_lower TEXT NOT NULL,
            labels     TEXT,
            summary    TEXT,
            attributes TEXT,
            created_at TEXT
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_nodes_graph_name
            ON nodes(graph_id, name_lower);
        CREATE INDEX IF NOT EXISTS idx_nodes_graph_id
            ON nodes(graph_id);

        CREATE TABLE IF NOT EXISTS edges (
            uuid             TEXT PRIMARY KEY,
            graph_id         TEXT NOT NULL REFERENCES graphs(graph_id) ON DELETE CASCADE,
            name             TEXT,
            fact             TEXT,
            fact_type        TEXT,
            source_node_uuid TEXT,
            target_node_uuid TEXT,
            attributes       TEXT,
            valid_at         TEXT,
            invalid_at       TEXT,
            expired_at       TEXT,
            created_at       TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_edges_conflict
            ON edges(graph_id, source_node_uuid, target_node_uuid, name, expired_at);
        CREATE INDEX IF NOT EXISTS idx_edges_graph_id
            ON edges(graph_id);

        CREATE TABLE IF NOT EXISTS episodes (
            uuid       TEXT PRIMARY KEY,
            graph_id   TEXT NOT NULL REFERENCES graphs(graph_id) ON DELETE CASCADE,
            data       TEXT,
            type       TEXT,
            processed  INTEGER DEFAULT 0,
            created_at TEXT
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
            uuid UNINDEXED,
            graph_id UNINDEXED,
            name,
            summary,
            tokenize="trigram"
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS edges_fts USING fts5(
            uuid UNINDEXED,
            graph_id UNINDEXED,
            fact,
            name,
            tokenize="trigram"
        );

        CREATE TRIGGER IF NOT EXISTS nodes_ai AFTER INSERT ON nodes BEGIN
            INSERT INTO nodes_fts(uuid, graph_id, name, summary)
            VALUES (new.uuid, new.graph_id, new.name, new.summary);
        END;

        CREATE TRIGGER IF NOT EXISTS nodes_au AFTER UPDATE ON nodes BEGIN
            DELETE FROM nodes_fts WHERE uuid = old.uuid;
            INSERT INTO nodes_fts(uuid, graph_id, name, summary)
            VALUES (new.uuid, new.graph_id, new.name, new.summary);
        END;

        CREATE TRIGGER IF NOT EXISTS nodes_ad AFTER DELETE ON nodes BEGIN
            DELETE FROM nodes_fts WHERE uuid = old.uuid;
        END;

        CREATE TRIGGER IF NOT EXISTS edges_ai AFTER INSERT ON edges BEGIN
            INSERT INTO edges_fts(uuid, graph_id, fact, name)
            VALUES (new.uuid, new.graph_id, new.fact, new.name);
        END;

        CREATE TRIGGER IF NOT EXISTS edges_au AFTER UPDATE ON edges BEGIN
            DELETE FROM edges_fts WHERE uuid = old.uuid;
            INSERT INTO edges_fts(uuid, graph_id, fact, name)
            VALUES (new.uuid, new.graph_id, new.fact, new.name);
        END;

        CREATE TRIGGER IF NOT EXISTS edges_ad AFTER DELETE ON edges BEGIN
            DELETE FROM edges_fts WHERE uuid = old.uuid;
        END;
    """)
    conn.commit()


# ─── Graph ────────────────────────────────────────────────────────────────────

def create_graph(graph_id: str, name: str = "", description: str = ""):
    with _lock:
        conn = _get_conn()
        conn.execute(
            "INSERT OR IGNORE INTO graphs(graph_id, name, description, created_at) VALUES(?,?,?,?)",
            (graph_id, name, description, _now())
        )
        conn.commit()


def delete_graph(graph_id: str):
    with _lock:
        conn = _get_conn()
        conn.execute("DELETE FROM graphs WHERE graph_id=?", (graph_id,))
        conn.commit()


def set_graph_ontology(graph_id: str, ontology: dict):
    with _lock:
        conn = _get_conn()
        conn.execute(
            "UPDATE graphs SET ontology=? WHERE graph_id=?",
            (json.dumps(ontology, ensure_ascii=False), graph_id)
        )
        conn.commit()


def get_graph_ontology(graph_id: str) -> Optional[dict]:
    conn = _get_conn()
    row = conn.execute("SELECT ontology FROM graphs WHERE graph_id=?", (graph_id,)).fetchone()
    if row and row["ontology"]:
        return json.loads(row["ontology"])
    return None


def get_graph_stats(graph_id: str) -> dict:
    """COUNT-based stats — much faster than fetching all rows."""
    conn = _get_conn()
    node_count = conn.execute(
        "SELECT COUNT(*) FROM nodes WHERE graph_id=?", (graph_id,)
    ).fetchone()[0]
    edge_count = conn.execute(
        "SELECT COUNT(*) FROM edges WHERE graph_id=?", (graph_id,)
    ).fetchone()[0]
    active_edge_count = conn.execute(
        "SELECT COUNT(*) FROM edges WHERE graph_id=? AND expired_at IS NULL",
        (graph_id,)
    ).fetchone()[0]
    ep_total = conn.execute(
        "SELECT COUNT(*) FROM episodes WHERE graph_id=?", (graph_id,)
    ).fetchone()[0]
    ep_done = conn.execute(
        "SELECT COUNT(*) FROM episodes WHERE graph_id=? AND processed=1", (graph_id,)
    ).fetchone()[0]
    return {
        "node_count": node_count,
        "edge_count": edge_count,
        "active_edge_count": active_edge_count,
        "episode_count": ep_total,
        "processed_episode_count": ep_done,
    }


# ─── Node merge helpers ───────────────────────────────────────────────────────

def _merge_labels(old: list, new: list) -> list:
    """Union of labels, preserving order, deduplicating."""
    seen: set = set()
    result = []
    for label in old + new:
        if label and label not in seen:
            seen.add(label)
            result.append(label)
    return result


def _merge_summaries(old: str, new: str, max_len: int = 400) -> str:
    """Accumulate summaries up to max_len.  New info appended; duplicates skipped."""
    old = (old or "").strip()
    new = (new or "").strip()
    if not new or new == old:
        return old
    if not old:
        return new[:max_len]
    combined = f"{old}；{new}"
    return combined[:max_len]


def _merge_attrs(old: dict, new: dict) -> dict:
    """Merge attribute dicts: existing keys preserved, new non-null values overwrite."""
    if not new:
        return old
    if not old:
        return new
    merged = dict(old)
    for k, v in new.items():
        if v is not None:
            merged[k] = v
    return merged


# ─── Nodes ────────────────────────────────────────────────────────────────────

def upsert_node(graph_id: str, name: str, labels: list, summary: str, attrs: dict) -> str:
    """按 (graph_id, name_lower) 去重。
    已存在时合并 labels（取并集）、summary（累加）、attributes（dict 合并），
    而非直接覆盖，保留跨 episode 积累的信息。
    """
    name_lower = name.lower().strip()
    with _lock:
        conn = _get_conn()
        row = conn.execute(
            "SELECT uuid, labels, summary, attributes FROM nodes WHERE graph_id=? AND name_lower=?",
            (graph_id, name_lower)
        ).fetchone()
        if row:
            node_uuid = row["uuid"]
            old_labels = json.loads(row["labels"]) if row["labels"] else []
            old_summary = row["summary"] or ""
            old_attrs = json.loads(row["attributes"]) if row["attributes"] else {}

            merged_labels = _merge_labels(old_labels, labels)
            merged_summary = _merge_summaries(old_summary, summary)
            merged_attrs = _merge_attrs(old_attrs, attrs)

            conn.execute(
                "UPDATE nodes SET labels=?, summary=?, attributes=? WHERE uuid=?",
                (json.dumps(merged_labels, ensure_ascii=False),
                 merged_summary,
                 json.dumps(merged_attrs, ensure_ascii=False),
                 node_uuid)
            )
        else:
            node_uuid = str(_uuid.uuid4())
            conn.execute(
                "INSERT INTO nodes(uuid, graph_id, name, name_lower, labels, summary, attributes, created_at)"
                " VALUES(?,?,?,?,?,?,?,?)",
                (node_uuid, graph_id, name, name_lower,
                 json.dumps(labels, ensure_ascii=False),
                 summary,
                 json.dumps(attrs, ensure_ascii=False),
                 _now())
            )
        conn.commit()
    return node_uuid


def get_node(uuid_: str) -> Optional[NodeResponse]:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM nodes WHERE uuid=?", (uuid_,)).fetchone()
    return _row_to_node(row) if row else None


def get_node_by_name(graph_id: str, name: str) -> Optional[NodeResponse]:
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM nodes WHERE graph_id=? AND name_lower=?",
        (graph_id, name.lower().strip())
    ).fetchone()
    return _row_to_node(row) if row else None


def get_nodes_page(graph_id: str, limit: int = 100, uuid_cursor: Optional[str] = None) -> list:
    conn = _get_conn()
    if uuid_cursor:
        rows = conn.execute(
            "SELECT * FROM nodes WHERE graph_id=? AND uuid>? ORDER BY uuid LIMIT ?",
            (graph_id, uuid_cursor, limit)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM nodes WHERE graph_id=? ORDER BY uuid LIMIT ?",
            (graph_id, limit)
        ).fetchall()
    return [_row_to_node(r) for r in rows]


def get_node_edges(node_uuid: str) -> list:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM edges WHERE source_node_uuid=? OR target_node_uuid=?",
        (node_uuid, node_uuid)
    ).fetchall()
    return [_row_to_edge(r) for r in rows]


def _row_to_node(row) -> NodeResponse:
    return NodeResponse(
        uuid_=row["uuid"],
        name=row["name"],
        labels=json.loads(row["labels"]) if row["labels"] else [],
        summary=row["summary"] or "",
        attributes=json.loads(row["attributes"]) if row["attributes"] else {},
        created_at=row["created_at"],
    )


# ─── Edges ────────────────────────────────────────────────────────────────────

def create_edge(
    graph_id: str,
    name: str,
    fact: str,
    source_node: NodeResponse,
    target_node: NodeResponse,
    fact_type: str = "",
    attrs: Optional[dict] = None,
) -> str:
    """创建边，并将相同 (src, tgt, relation) 的旧边标记为 expired/invalid。

    这实现了 Zep Cloud 的时间性图谱功能子集：
    - 新边获得 valid_at = now（事实生效时间）
    - 旧的同关系边被设为 expired_at = invalid_at = now（历史事实）
    → PanoramaSearch 可以正确区分当前事实与历史事实
    """
    edge_uuid = str(_uuid.uuid4())
    now = _now()
    with _lock:
        conn = _get_conn()
        # 将已有的相同关系边标记为历史
        conn.execute(
            "UPDATE edges SET expired_at=?, invalid_at=? "
            "WHERE graph_id=? AND source_node_uuid=? AND target_node_uuid=? "
            "  AND name=? AND expired_at IS NULL AND invalid_at IS NULL",
            (now, now, graph_id, source_node.uuid_, target_node.uuid_, name)
        )
        conn.execute(
            "INSERT INTO edges(uuid, graph_id, name, fact, fact_type, source_node_uuid,"
            " target_node_uuid, attributes, valid_at, created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (edge_uuid, graph_id, name, fact, fact_type or name,
             source_node.uuid_, target_node.uuid_,
             json.dumps(attrs or {}, ensure_ascii=False),
             now, now)  # valid_at = created_at = now
        )
        conn.commit()
    return edge_uuid


def get_edges_page(graph_id: str, limit: int = 100, uuid_cursor: Optional[str] = None) -> list:
    conn = _get_conn()
    if uuid_cursor:
        rows = conn.execute(
            "SELECT * FROM edges WHERE graph_id=? AND uuid>? ORDER BY uuid LIMIT ?",
            (graph_id, uuid_cursor, limit)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM edges WHERE graph_id=? ORDER BY uuid LIMIT ?",
            (graph_id, limit)
        ).fetchall()
    return [_row_to_edge(r) for r in rows]


def _row_to_edge(row) -> EdgeResponse:
    return EdgeResponse(
        uuid_=row["uuid"],
        graph_id=row["graph_id"],
        name=row["name"] or "",
        fact=row["fact"] or "",
        fact_type=row["fact_type"] or "",
        source_node_uuid=row["source_node_uuid"] or "",
        target_node_uuid=row["target_node_uuid"] or "",
        attributes=json.loads(row["attributes"]) if row["attributes"] else {},
        valid_at=row["valid_at"],
        invalid_at=row["invalid_at"],
        expired_at=row["expired_at"],
        created_at=row["created_at"],
    )


# ─── Episodes ─────────────────────────────────────────────────────────────────

def create_episode(uuid_: str, graph_id: str, data: str, type_: str = "text") -> str:
    with _lock:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO episodes(uuid, graph_id, data, type, processed, created_at) VALUES(?,?,?,?,0,?)",
            (uuid_, graph_id, data, type_, _now())
        )
        conn.commit()
    return uuid_


def mark_episode_processed(uuid_: str):
    with _lock:
        conn = _get_conn()
        conn.execute("UPDATE episodes SET processed=1 WHERE uuid=?", (uuid_,))
        conn.commit()


def get_episode(uuid_: str) -> Optional[EpisodeResponse]:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM episodes WHERE uuid=?", (uuid_,)).fetchone()
    if not row:
        return None
    return EpisodeResponse(
        uuid_=row["uuid"],
        graph_id=row["graph_id"],
        data=row["data"] or "",
        type=row["type"] or "text",
        processed=bool(row["processed"]),
        created_at=row["created_at"],
    )


# ─── Token extraction (shared by FTS and keyword fallback) ────────────────────

def _extract_tokens(query: str) -> list:
    """从查询字符串中提取有意义的搜索 token，按长度降序排列。

    策略：
    1. ASCII 词（英文实体名如 "Alice"，至少 2 字符）
    2. 中文：在常见单字虚词（的了和与及或在是于以为被）及非 CJK 字符处切分，
       保留 ≥2 字符的片段（每片段是潜在的名词或名词短语）
    """
    query = query.strip()
    if not query:
        return []

    ascii_tokens = _re.findall(r'[A-Za-z0-9_]{2,}', query)

    cjk_text = _re.sub(r'[A-Za-z0-9_\s]', '', query)
    cjk_segments = _re.split(
        r'[^\u4e00-\u9fff\u3400-\u4dbf]|[的了和与及或在是于以为被]',
        cjk_text
    )
    cjk_tokens = [s for s in cjk_segments if len(s) >= 2]

    seen: set = set()
    tokens: list = []
    for t in ascii_tokens + cjk_tokens:
        t = t.strip()
        if t and t not in seen:
            seen.add(t)
            tokens.append(t)
    tokens.sort(key=len, reverse=True)
    return tokens


# ─── FTS Search ───────────────────────────────────────────────────────────────

def search_nodes_fts(graph_ids: list, query: str, limit: int = 20) -> list:
    if not graph_ids or not query:
        return []
    conn = _get_conn()
    placeholders = ",".join("?" * len(graph_ids))
    rows = conn.execute(
        f"""
        SELECT n.* FROM nodes_fts f
        JOIN nodes n ON n.uuid = f.uuid
        WHERE f MATCH ?
          AND f.graph_id IN ({placeholders})
        ORDER BY rank
        LIMIT ?
        """,
        [_fts_escape(query)] + list(graph_ids) + [limit]
    ).fetchall()
    return [_row_to_node(r) for r in rows]


def search_edges_fts(graph_ids: list, query: str, limit: int = 20) -> list:
    if not graph_ids or not query:
        return []
    conn = _get_conn()
    placeholders = ",".join("?" * len(graph_ids))
    rows = conn.execute(
        f"""
        SELECT e.* FROM edges_fts f
        JOIN edges e ON e.uuid = f.uuid
        WHERE f MATCH ?
          AND f.graph_id IN ({placeholders})
          AND e.expired_at IS NULL
        ORDER BY rank
        LIMIT ?
        """,
        [_fts_escape(query)] + list(graph_ids) + [limit]
    ).fetchall()
    return [_row_to_edge(r) for r in rows]


# ─── Keyword fallback search (LIKE scan, used when FTS5 returns nothing) ──────

def search_nodes_keyword(graph_ids: list, query: str, limit: int = 20) -> list:
    """LIKE-based keyword scan — fallback when FTS5 yields no results.
    Uses the top 3 extracted tokens with OR; each token matched against
    both name and summary columns.
    """
    tokens = _extract_tokens(query)[:3]
    if not tokens:
        return []
    conn = _get_conn()
    placeholders_g = ",".join("?" * len(graph_ids))
    conditions = []
    params: list = []
    for t in tokens:
        kw = f"%{t}%"
        conditions.append("(name LIKE ? OR summary LIKE ?)")
        params.extend([kw, kw])
    where = " OR ".join(conditions)
    rows = conn.execute(
        f"SELECT * FROM nodes WHERE graph_id IN ({placeholders_g}) AND ({where}) LIMIT ?",
        list(graph_ids) + params + [limit]
    ).fetchall()
    return [_row_to_node(r) for r in rows]


def search_edges_keyword(graph_ids: list, query: str, limit: int = 20) -> list:
    """LIKE-based keyword scan — fallback when FTS5 yields no results.
    Uses the top 3 extracted tokens with OR; each token matched against
    fact and name columns.
    """
    tokens = _extract_tokens(query)[:3]
    if not tokens:
        return []
    conn = _get_conn()
    placeholders_g = ",".join("?" * len(graph_ids))
    conditions = []
    params: list = []
    for t in tokens:
        kw = f"%{t}%"
        conditions.append("(fact LIKE ? OR name LIKE ?)")
        params.extend([kw, kw])
    where = " OR ".join(conditions)
    rows = conn.execute(
        f"SELECT * FROM edges WHERE graph_id IN ({placeholders_g}) AND expired_at IS NULL AND ({where}) LIMIT ?",
        list(graph_ids) + params + [limit]
    ).fetchall()
    return [_row_to_edge(r) for r in rows]


def _fts_escape(query: str) -> str:
    """将查询字符串转换为 FTS5 trigram MATCH 表达式。

    FTS5 trigram tokenizer 做子串匹配：`"word"` 匹配所有含 "word" 子串的记录。

    短查询（≤6字符）直接用整体短语，适用于实体名等精确检索。
    长查询（自然语言）拆分为关键词 OR 组合，提升召回率——避免把整句话
    当成一个子串去搜导致命中率为零的问题（这是 Zep Cloud 语义搜索退化的
    最主要场景，例如 "关于Alice的所有信息、活动、事件、关系和背景"）。
    """
    query = query.strip()
    if not query:
        return '""'

    # 短查询（纯实体名等）：整体短语搜索足够
    if len(query) <= 6:
        return f'"{query.replace(chr(34), chr(34) * 2)}"'

    # 长查询：提取有意义的 token，用 OR 组合
    tokens = _extract_tokens(query)[:8]  # 最多 8 个 token

    if not tokens:
        # 兜底：取前 12 字符做短语搜索
        short = query[:12]
        return f'"{short.replace(chr(34), chr(34) * 2)}"'

    return ' OR '.join(f'"{t.replace(chr(34), chr(34) * 2)}"' for t in tokens)

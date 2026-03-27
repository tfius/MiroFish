"""
local_zep SQLite storage layer
WAL mode + FTS5 trigram index, thread-safe writes

Improvements:
- upsert_node merges labels/summary/attributes (no overwrites)
- create_edge sets valid_at and marks old edges with the same relation as expired/invalid
  → enables PanoramaSearch to distinguish current facts from historical facts
- search_*_keyword keyword fallback search (used when FTS5 returns no results)
- get_graph_stats uses COUNT for efficient statistics
"""

import json
import os
import re as _re
import sqlite3
import threading
import uuid as _uuid
from datetime import datetime, timezone
from typing import Any, Optional

try:
    import sqlite_vec
    _SQLITE_VEC_AVAILABLE = True
except ImportError:
    _SQLITE_VEC_AVAILABLE = False

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
        _migrate_schema(conn)


def _get_conn() -> sqlite3.Connection:
    """Each thread maintains its own connection (WAL allows multiple readers, one writer)."""
    if not hasattr(_local, "conn") or _local.conn is None:
        conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        if _SQLITE_VEC_AVAILABLE:
            try:
                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
                conn.enable_load_extension(False)
            except AttributeError:
                # SQLite was compiled without extension-loading support (common on some Linux distros)
                globals()["_SQLITE_VEC_AVAILABLE"] = False
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


def _migrate_schema(conn: sqlite3.Connection):
    """Add columns introduced after initial schema creation (idempotent)."""
    for ddl in (
        "ALTER TABLE nodes ADD COLUMN embedding BLOB",
        "ALTER TABLE edges ADD COLUMN embedding BLOB",
    ):
        try:
            conn.execute(ddl)
            conn.commit()
        except sqlite3.OperationalError as e:
            if "duplicate column name" not in str(e).lower():
                raise  # re-raise unexpected errors (read-only DB, corruption, etc.)


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
        cur = conn.execute(
            "UPDATE graphs SET ontology=? WHERE graph_id=?",
            (json.dumps(ontology, ensure_ascii=False), graph_id)
        )
        if cur.rowcount == 0:
            from ..utils.logger import get_logger as _get_logger
            _get_logger("mirofish.local_zep.db").warning(
                f"set_graph_ontology: graph '{graph_id}' not found — ontology not saved. "
                "Call create_graph() first."
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
    combined = f"{old}. {new}"
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
    """Deduplicates by (graph_id, name_lower).
    When a node already exists, merges labels (union), summary (accumulate), and attributes
    (dict merge) rather than overwriting, preserving information accumulated across episodes.
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
        graph_id=row["graph_id"],
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
    """Creates an edge and marks existing edges with the same (src, tgt, relation) as expired/invalid.

    This implements a subset of Zep Cloud's temporal graph functionality:
    - The new edge receives valid_at = now (fact effective time)
    - Old edges with the same relation are set to expired_at = invalid_at = now (historical facts)
    → PanoramaSearch can correctly distinguish current facts from historical facts.
    """
    edge_uuid = str(_uuid.uuid4())
    now = _now()
    with _lock:
        conn = _get_conn()
        # Mark existing edges with the same relation as historical
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
    """Extract meaningful search tokens from the query string, sorted by descending length.

    Strategy:
    1. ASCII words (English entity names such as "Alice", at least 2 characters)
    2. Chinese: split on common single-character function words (的了和与及或在是于以为被)
       and non-CJK characters; keep segments of ≥2 characters (each segment is a
       potential noun or noun phrase).
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
        SELECT n.* FROM nodes_fts
        JOIN nodes n ON n.uuid = nodes_fts.uuid
        WHERE nodes_fts MATCH ?
          AND nodes_fts.graph_id IN ({placeholders})
        ORDER BY nodes_fts.rank
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
        SELECT e.* FROM edges_fts
        JOIN edges e ON e.uuid = edges_fts.uuid
        WHERE edges_fts MATCH ?
          AND edges_fts.graph_id IN ({placeholders})
          AND e.expired_at IS NULL
        ORDER BY edges_fts.rank
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


# ─── Embedding storage ────────────────────────────────────────────────────────

def store_node_embedding(node_uuid: str, embedding: bytes):
    """Store a pre-computed float32 embedding blob for a node."""
    if not embedding:
        return
    with _lock:
        conn = _get_conn()
        conn.execute("UPDATE nodes SET embedding=? WHERE uuid=?", (embedding, node_uuid))
        conn.commit()


def store_edge_embedding(edge_uuid: str, embedding: bytes):
    """Store a pre-computed float32 embedding blob for an edge."""
    if not embedding:
        return
    with _lock:
        conn = _get_conn()
        conn.execute("UPDATE edges SET embedding=? WHERE uuid=?", (embedding, edge_uuid))
        conn.commit()


# ─── Vector search (cosine similarity via sqlite-vec scalar function) ─────────

def search_nodes_vec(graph_ids: list, query_embedding: bytes, limit: int = 20) -> list:
    """KNN search over node embeddings using vec_distance_cosine.

    Falls back to empty list if sqlite-vec is unavailable, no query embedding, or on error.
    Uses a full scan (suitable for graphs up to ~50k nodes).
    """
    if not _SQLITE_VEC_AVAILABLE or not graph_ids or not query_embedding:
        return []
    try:
        conn = _get_conn()
        placeholders = ",".join("?" * len(graph_ids))
        rows = conn.execute(
            f"""
            SELECT *, vec_distance_cosine(embedding, ?) AS dist
            FROM nodes
            WHERE graph_id IN ({placeholders})
              AND embedding IS NOT NULL
            ORDER BY dist
            LIMIT ?
            """,
            [query_embedding] + list(graph_ids) + [limit]
        ).fetchall()
        return [_row_to_node(r) for r in rows]
    except Exception:
        return []


def search_edges_vec(graph_ids: list, query_embedding: bytes, limit: int = 20) -> list:
    """KNN search over edge embeddings using vec_distance_cosine.

    Only returns active edges (expired_at IS NULL).
    Falls back to empty list on any error (e.g. dimension mismatch from a model change).
    """
    if not _SQLITE_VEC_AVAILABLE or not graph_ids or not query_embedding:
        return []
    try:
        conn = _get_conn()
        placeholders = ",".join("?" * len(graph_ids))
        rows = conn.execute(
            f"""
            SELECT *, vec_distance_cosine(embedding, ?) AS dist
            FROM edges
            WHERE graph_id IN ({placeholders})
              AND expired_at IS NULL
              AND embedding IS NOT NULL
            ORDER BY dist
            LIMIT ?
            """,
            [query_embedding] + list(graph_ids) + [limit]
        ).fetchall()
        return [_row_to_edge(r) for r in rows]
    except Exception:
        return []


def _fts_escape(query: str) -> str:
    """Convert a query string into an FTS5 trigram MATCH expression.

    The FTS5 trigram tokenizer does substring matching: `"word"` matches all records
    containing the substring "word".

    Short queries (≤6 characters) are used as a single phrase, suitable for precise
    lookups such as entity names.
    Long queries (natural language) are split into keyword OR combinations to improve
    recall — this avoids the zero-hit problem that occurs when the entire sentence is
    treated as one substring (the most common degraded-search scenario with Zep Cloud
    semantic search, e.g. "all information, activities, events, relationships and
    background about Alice").
    """
    query = query.strip()
    if not query:
        return '""'

    # Short query (pure entity name, etc.): a single phrase search is sufficient
    if len(query) <= 6:
        return f'"{query.replace(chr(34), chr(34) * 2)}"'

    # Long query: extract meaningful tokens and combine with OR
    tokens = _extract_tokens(query)[:8]  # at most 8 tokens

    if not tokens:
        # Fallback: use the first 12 characters as a phrase search
        short = query[:12]
        return f'"{short.replace(chr(34), chr(34) * 2)}"'

    return ' OR '.join(f'"{t.replace(chr(34), chr(34) * 2)}"' for t in tokens)

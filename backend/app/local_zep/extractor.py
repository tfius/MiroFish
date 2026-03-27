"""
local_zep entity/relationship extractor
Reuses the existing LLMClient to extract entities and relationships from text
via a small local model.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import threading

from ..config import Config
from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger
from . import db
from . import embedder

logger = get_logger("mirofish.local_zep.extractor")

_executor: Optional[ThreadPoolExecutor] = None
_executor_lock = threading.Lock()
_llm: Optional[LLMClient] = None
_llm_lock = threading.Lock()


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        with _executor_lock:
            if _executor is None:
                workers = getattr(Config, "LLM_EXTRACT_WORKERS", 2)
                _executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="local_zep")
    return _executor


def _get_llm() -> LLMClient:
    global _llm
    if _llm is None:
        with _llm_lock:
            if _llm is None:
                _llm = LLMClient(
                    api_key=Config.LLM_EXTRACT_API_KEY,
                    base_url=Config.LLM_EXTRACT_BASE_URL,
                    model=Config.LLM_EXTRACT_MODEL_NAME,
                )
    return _llm


_SYSTEM_PROMPT = """You are a knowledge graph construction assistant. Extract entities and relationships from the given text and return them in JSON format.

Output format (strict JSON):
{
  "entities": [
    {"name": "entity name", "type": "entity type", "summary": "brief description", "attributes": {}}
  ],
  "relationships": [
    {"source": "source entity name", "relation": "relation name", "target": "target entity name", "fact": "a single sentence describing this relationship"}
  ]
}

Rules:
- Normalise entity names (remove extra whitespace, unify capitalisation)
- Only extract entities and relationships explicitly mentioned in the text
- Keep summaries concise, no more than 50 words
- Use a verb or verb phrase for relation names
- If there are no entities or relationships, return an empty array for the corresponding field
- Return only JSON, no other text"""


def _build_prompt_with_ontology(text: str, ontology: Optional[dict]) -> list:
    system = _SYSTEM_PROMPT
    if ontology:
        entity_types = [e.get("name", "") for e in ontology.get("entity_types", [])]
        edge_types = [e.get("name", "") for e in ontology.get("edge_types", [])]
        hints = []
        if entity_types:
            hints.append(f"Prefer these entity types: {', '.join(entity_types)}")
        if edge_types:
            hints.append(f"Prefer these relation types: {', '.join(edge_types)}")
        if hints:
            system = system + "\n\nHints:\n" + "\n".join(hints)

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Please extract entities and relationships from the following text:\n\n{text}"},
    ]


def extract(text: str, ontology: Optional[dict] = None) -> tuple[list, list]:
    """
    Extract entities and relationships from text.
    Returns (entities, relationships); returns ([], []) on failure.
    """
    try:
        llm = _get_llm()
        messages = _build_prompt_with_ontology(text, ontology)
        result = llm.chat_json(messages, temperature=0.1, max_tokens=2048)
        entities = result.get("entities", []) or []
        rels = result.get("relationships", []) or []
        return entities, rels
    except Exception as e:
        logger.warning(f"Entity extraction failed: {e}")
        return [], []


def submit_extraction(ep_uuid: str, graph_id: str, data: str):
    """Submit a background extraction task."""
    _get_executor().submit(_run_extraction, ep_uuid, graph_id, data)


def _run_extraction(ep_uuid: str, graph_id: str, data: str):
    """Background extraction and database write."""
    try:
        ontology = db.get_graph_ontology(graph_id)
        entities, rels = extract(data, ontology)

        # Write nodes; cache name_lower → NodeResponse for reuse when writing edges
        node_cache: dict[str, object] = {}
        for e in entities:
            name = (e.get("name") or "").strip()
            if not name:
                continue
            try:
                db.upsert_node(
                    graph_id=graph_id,
                    name=name,
                    labels=[e.get("type", "Entity")],
                    summary=e.get("summary", ""),
                    attrs=e.get("attributes", {}),
                )
                node = db.get_node_by_name(graph_id, name)
                if node:
                    node_cache[name.lower()] = node
                    # Use the merged summary from the DB (not the episode snippet)
                    # so the embedding always represents the full accumulated knowledge.
                    emb = embedder.encode(f"{name}. {node.summary}")
                    db.store_node_embedding(node.uuid_, emb)
            except Exception as ex:
                logger.warning(f"Failed to write node ({name}): {ex}")

        # Write edges; prefer the cache, fall back to a DB query when not found
        for r in rels:
            src_name = (r.get("source") or "").strip()
            tgt_name = (r.get("target") or "").strip()
            relation = (r.get("relation") or "").strip()
            fact = (r.get("fact") or "").strip()
            if not (src_name and tgt_name and relation):
                continue
            try:
                src = node_cache.get(src_name.lower()) or db.get_node_by_name(graph_id, src_name)
                tgt = node_cache.get(tgt_name.lower()) or db.get_node_by_name(graph_id, tgt_name)
                if src and tgt:
                    edge_uuid = db.create_edge(graph_id, relation, fact or relation, src, tgt)
                    emb = embedder.encode(fact or relation)
                    db.store_edge_embedding(edge_uuid, emb)
            except Exception as ex:
                logger.warning(f"Failed to write edge ({src_name}-{relation}->{tgt_name}): {ex}")

        logger.debug(
            f"Extraction complete: ep={ep_uuid[:8]}, entities={len(entities)}, rels={len(rels)}"
        )
    except Exception as e:
        logger.warning(f"Background extraction error ep={ep_uuid[:8]}: {e}")
    finally:
        try:
            db.mark_episode_processed(ep_uuid)
        except Exception as mark_err:
            logger.error(f"mark_episode_processed failed ep={ep_uuid[:8]}: {mark_err}")

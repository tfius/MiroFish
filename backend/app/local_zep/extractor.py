"""
local_zep 实体/关系提取器
复用现有 LLMClient，通过小型本地模型从文本中提取实体和关系
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import threading

from ..config import Config
from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger
from . import db

logger = get_logger("mirofish.local_zep.extractor")

_executor: Optional[ThreadPoolExecutor] = None
_executor_lock = threading.Lock()


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        with _executor_lock:
            if _executor is None:
                workers = getattr(Config, "LLM_EXTRACT_WORKERS", 2)
                _executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="local_zep")
    return _executor

_llm: Optional[LLMClient] = None


def _get_llm() -> LLMClient:
    global _llm
    if _llm is None:
        _llm = LLMClient(
            api_key=Config.LLM_EXTRACT_API_KEY,
            base_url=Config.LLM_EXTRACT_BASE_URL,
            model=Config.LLM_EXTRACT_MODEL_NAME,
        )
    return _llm


_SYSTEM_PROMPT = """你是一个知识图谱构建助手。从给定文本中提取实体和关系，以JSON格式返回。

输出格式（严格JSON）：
{
  "entities": [
    {"name": "实体名称", "type": "实体类型", "summary": "简短描述", "attributes": {}}
  ],
  "relationships": [
    {"source": "源实体名称", "relation": "关系名称", "target": "目标实体名称", "fact": "描述这个关系的一句话"}
  ]
}

规则：
- 实体名称要规范化（去除多余空格，统一大小写）
- 只提取文本中明确提到的实体和关系
- summary 简洁，不超过50字
- 关系名称用动词或动词短语
- 如果没有实体或关系，对应字段返回空数组
- 只返回JSON，不要其他文字"""


def _build_prompt_with_ontology(text: str, ontology: Optional[dict]) -> list:
    system = _SYSTEM_PROMPT
    if ontology:
        entity_types = [e.get("name", "") for e in ontology.get("entity_types", [])]
        edge_types = [e.get("name", "") for e in ontology.get("edge_types", [])]
        hints = []
        if entity_types:
            hints.append(f"优先使用这些实体类型: {', '.join(entity_types)}")
        if edge_types:
            hints.append(f"优先使用这些关系类型: {', '.join(edge_types)}")
        if hints:
            system = system + "\n\n提示：\n" + "\n".join(hints)

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"请从以下文本中提取实体和关系：\n\n{text}"},
    ]


def extract(text: str, ontology: Optional[dict] = None) -> tuple[list, list]:
    """
    从文本提取实体和关系。
    返回 (entities, relationships)，失败时返回 ([], [])。
    """
    try:
        llm = _get_llm()
        messages = _build_prompt_with_ontology(text, ontology)
        result = llm.chat_json(messages, temperature=0.1, max_tokens=2048)
        entities = result.get("entities", []) or []
        rels = result.get("relationships", []) or []
        return entities, rels
    except Exception as e:
        logger.warning(f"实体提取失败: {e}")
        return [], []


def submit_extraction(ep_uuid: str, graph_id: str, data: str):
    """提交后台提取任务"""
    _get_executor().submit(_run_extraction, ep_uuid, graph_id, data)


def _run_extraction(ep_uuid: str, graph_id: str, data: str):
    """后台提取并写入数据库"""
    try:
        ontology = db.get_graph_ontology(graph_id)
        entities, rels = extract(data, ontology)

        # 写入节点，缓存 name_lower → NodeResponse 供边写入复用
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
            except Exception as ex:
                logger.debug(f"写入节点失败 ({name}): {ex}")

        # 写入边，优先用缓存，缺失时回落到 DB 查询
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
                    db.create_edge(graph_id, relation, fact or relation, src, tgt)
            except Exception as ex:
                logger.debug(f"写入边失败 ({src_name}-{relation}->{tgt_name}): {ex}")

        logger.debug(
            f"提取完成: ep={ep_uuid[:8]}, entities={len(entities)}, rels={len(rels)}"
        )
    except Exception as e:
        logger.warning(f"后台提取异常 ep={ep_uuid[:8]}: {e}")
    finally:
        try:
            db.mark_episode_processed(ep_uuid)
        except Exception as mark_err:
            logger.error(f"mark_episode_processed 失败 ep={ep_uuid[:8]}: {mark_err}")

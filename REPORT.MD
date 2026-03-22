# MiroFish Architecture Report

## Overview

MiroFish is a social media simulation research platform. Given a set of source documents, it:

1. Builds a knowledge graph from them
2. Generates agent personas (Oasis profiles) grounded in that graph
3. Runs a multi-agent social media simulation (OASIS) across two platforms (Reddit + Twitter) in parallel
4. Updates the knowledge graph in real-time as agents act
5. Generates a structured research report using a ReACT agent that queries the graph

The system is entirely self-hosted — no cloud APIs are required. All LLM calls go to a local OpenAI-compatible server (LM Studio).

---

## 8-Stage Pipeline

```
Documents
   │
   ▼
[1] Ontology Generation      ─── LLM (large model, 16k tokens)
   │  Infer entity types + edge types from document content
   │
   ▼
[2] Graph Build              ─── LLM (extraction model)
   │  Chunk documents → extract entities/relations → upsert into SQLite
   │  graph (local_zep)
   │
   ▼
[3] Agent Persona Generation ─── LLM (large model)
   │  Query graph → generate N agent profiles (OASIS format)
   │  Each profile: name, bio, personality, interests, demographics
   │
   ▼
[4] Simulation Configuration
   │  User sets: platform (Reddit/Twitter/both), post count, agent count,
   │  topic, number of rounds
   │
   ▼
[5] OASIS Simulation         ─── External subprocess (camel-oasis)
   │  Two parallel processes: one for Reddit, one for Twitter
   │  Agents read/write posts, comments, likes
   │  Each action is fed back into the knowledge graph
   │
   ▼
[6] Graph Update (real-time)
   │  ZepGraphMemoryUpdater receives action batches from OASIS
   │  Extracts entities from each action text → upserts into graph
   │
   ▼
[7] Report Generation        ─── ReACT agent (LLM loop)
   │  Analyst agent uses 4 tools to query the graph
   │  Produces a structured research report
   │
   ▼
[8] Interactive Q&A
      User can ask questions; answered by the same ReACT agent
```

---

## Component Deep Dives

### Stage 1: Ontology Generation (`services/ontology_generator.py`)

- Takes the first 2000 characters of each uploaded document as context
- Sends a single LLM call (large model, `max_tokens=64000`) asking for entity types and edge types
- Returns a typed ontology: `{entity_types: [{name, description}], edge_types: [{name, description}]}`
- This ontology is stored per-graph in SQLite and later used as extraction hints

### Stage 2: Graph Build (`services/graph_builder.py`)

- Splits documents into chunks using `TextProcessor.split_text()`
- For each chunk, calls `local_zep.graph.add()` which:
  - Saves the chunk as an episode in SQLite
  - Submits an async extraction task to a thread pool (2 workers)
- Extraction worker (`local_zep/extractor.py`):
  - Loads ontology hints from DB
  - Calls extraction LLM with a JSON-output prompt
  - Parses `{entities: [...], relationships: [...]}`
  - Upserts nodes (dedup by `name_lower`) and edges into SQLite
  - Marks episode as processed

### Stage 3: Oasis Profile Generation (`services/oasis_profile_generator.py`)

- Queries graph for existing nodes (entities)
- Builds an LLM prompt describing the graph context
- Asks the LLM to generate N agent profiles in OASIS JSON format
- Profiles written to `backend/data/oasis_profiles.json`

### Stage 5: OASIS Simulation (`services/oasis_runner.py`)

- Launches `camel-oasis` as a subprocess
- If both platforms selected: launches two separate subprocesses in parallel
  - Reddit subprocess: posts + comments on Reddit-style posts
  - Twitter subprocess: tweets + replies on Twitter-style posts
- Each subprocess runs the simulation for a configured number of rounds
- Output: action logs and post/comment data streamed back via stdout

### Stage 6: Real-Time Graph Updates (`services/zep_graph_memory_updater.py`)

- Receives batches of OASIS actions (agent utterances, posts, comments)
- For each batch: calls `local_zep.graph.add_batch()`
- Extraction LLM processes each action text asynchronously
- New entities and relationships from simulation behavior are added to the graph
- This creates a feedback loop: document knowledge + simulation behavior = richer graph

### Stage 7: Report Generation (`services/report_generator.py`)

ReACT agent loop:
1. LLM reasons about what to investigate
2. LLM selects one of 4 tools and emits a tool call
3. Tool executes, result fed back to LLM
4. Loop until LLM emits `FINAL_REPORT`

**4 ReACT tools:**

| Tool | What it does |
|------|-------------|
| `InsightForge` | Semantic edge search — finds relationship facts by keyword |
| `PanoramaSearch` | Node + edge stats query — "what entities exist for this topic?" |
| `QuickSearch` | Fast node name search by keyword |
| `InterviewAgents` | Ask a question to simulated agents; runs mini sub-simulation |

### Stage 8: Interactive Q&A (`services/zep_tools.py`)

- Same 4 tools, same ReACT loop
- User question is the starting prompt
- Answer streamed back to frontend via SSE

---

## Knowledge Graph

**Storage:** SQLite (`backend/data/local_zep.db`) with WAL mode + FTS5 trigram index

**Schema:**
```
graphs   — graph_id, name, description, ontology JSON
nodes    — uuid, graph_id, name, name_lower, labels JSON, summary, attributes JSON
edges    — uuid, graph_id, name (relation), fact, source_node_uuid, target_node_uuid,
           valid_at, invalid_at, expired_at
episodes — uuid, graph_id, data (raw text chunk), processed (0/1)

nodes_fts — FTS5 virtual table (trigram tokenizer) → name, summary
edges_fts — FTS5 virtual table (trigram tokenizer) → fact, name
```

**Key behaviors:**
- Nodes deduped by `(graph_id, name_lower)` — same entity mentioned multiple times = single node
- Edges use temporal invalidation: conflicting edges (same source + relation + target) get `expired_at` set; only current edges are returned in search
- FTS5 trigram tokenizer handles Chinese named entities naturally (substring matching)
- Keyword LIKE fallback when FTS returns empty results

---

## LLM Call Inventory

| Stage | Model | Purpose | Tokens |
|-------|-------|---------|--------|
| Ontology generation | Large (e.g., Qwen3-35b) | Infer entity/edge type schema from documents | 16k max |
| Profile generation | Large | Generate OASIS agent personas | 4k max |
| Entity extraction | Small (e.g., Qwen3.5-2b) | Extract entities + relations from text chunks | 2k max |
| Report generation | Large | ReACT reasoning + tool selection | 4k per step |
| InterviewAgents tool | Large | Run mini agent simulation | 4k max |
| Interactive Q&A | Large | Same ReACT loop | 4k per step |

**All calls:** Local LM Studio server (OpenAI-compatible API). No internet required.

**Two separate model endpoints configurable:**
- `LLM_*` — main/large model
- `LLM_EXTRACT_*` — small extraction model (can be same server, different model)

---

## File System Persistence

```
backend/
├── data/
│   ├── local_zep.db          — knowledge graph (SQLite)
│   ├── oasis_profiles.json   — generated agent personas
│   ├── uploads/              — uploaded source documents
│   └── simulation_output/    — OASIS subprocess output logs
└── app/
    ├── local_zep/            — self-hosted Zep replacement
    │   ├── __init__.py       — exports: Zep, EpisodeData, EntityEdgeSourceTarget, InternalServerError
    │   ├── client.py         — LocalZep class (drop-in for zep_cloud.client.Zep)
    │   ├── db.py             — all SQLite operations
    │   ├── extractor.py      — async LLM entity extraction
    │   ├── models.py         — NodeResponse, EdgeResponse, EpisodeResponse dataclasses
    │   ├── ontology.py       — EntityModel, EdgeModel, EntityText stubs
    │   └── search.py         — FTS5 + keyword fallback search
    ├── services/
    │   ├── ontology_generator.py
    │   ├── graph_builder.py
    │   ├── oasis_profile_generator.py
    │   ├── oasis_runner.py
    │   ├── zep_graph_memory_updater.py
    │   ├── zep_entity_reader.py
    │   ├── zep_tools.py
    │   └── report_generator.py
    └── utils/
        ├── llm_client.py     — OpenAI-compatible client (strips <think> tags, markdown fences)
        ├── zep_paging.py     — cursor-based pagination helpers
        └── text_processor.py — document chunking
```

---

## Dependency Graph

```
graph_builder.py
zep_graph_memory_updater.py    ─┐
zep_entity_reader.py            ├──► local_zep/client.py
zep_tools.py                    │         │
oasis_profile_generator.py     ─┘         ├──► db.py (SQLite)
                                          ├──► extractor.py ──► llm_client.py
                                          └──► search.py ───► db.py

report_generator.py ──► zep_tools.py ──► local_zep/client.py
ontology_generator.py ──► llm_client.py
oasis_runner.py ──► camel-oasis (subprocess)
```

---

## External Dependencies

| Dependency | Purpose | Notes |
|------------|---------|-------|
| LM Studio | LLM inference server | OpenAI-compatible API at localhost |
| camel-oasis | Social media simulation engine | Launched as subprocess |
| FastAPI | Backend HTTP/SSE server | |
| SQLite (built-in) | Knowledge graph storage | WAL + FTS5 trigram |
| Vue 3 + Vite | Frontend | |

**No cloud services required.** The original architecture used Zep Cloud (paid knowledge graph service); this has been replaced by the self-hosted `local_zep` module.

---

## Data Flow: Document to Report

```
PDF/TXT upload
    │
    ▼ (file stored in uploads/)
Ontology LLM call
    │ entity types, edge types
    ▼ (stored in graphs.ontology)
Document chunking (TextProcessor)
    │ ~500-char chunks
    ▼
Async extraction (2-worker thread pool)
    │ LLM: {entities, relationships}
    ▼ (upserted to nodes + edges tables)
Agent profile LLM call
    │ reads nodes from graph as context
    ▼ (written to oasis_profiles.json)
OASIS subprocess(es)
    │ agent actions as text
    ▼
Real-time graph updates (ZepGraphMemoryUpdater)
    │ new entities from simulation added to graph
    ▼
ReACT report agent
    │ queries graph via 4 tools
    ▼
Structured report (streamed to frontend via SSE)
```

---

## Notable Design Decisions

**1. Two-world parallel simulation**
Reddit and Twitter run as independent subprocesses. This allows comparing how the same agents behave on different platforms, and how platform affordances (tweet length limits vs. long-form Reddit posts) affect discussion dynamics.

**2. Graph as shared memory**
The knowledge graph is not static. It grows during simulation as agents discuss new topics. This means the report agent sees not only what was in the source documents, but also what emerged from agent interaction — emergent knowledge.

**3. Async extraction pipeline**
Entity extraction is decoupled from episode ingestion. The `graph.add()` call returns immediately after writing the raw text to SQLite; extraction happens in background threads. This keeps the graph build pipeline fast even when the extraction LLM is slow.

**4. Reasoning model compatibility**
The LLM client strips `<think>...</think>` blocks and markdown code fences from all responses. The ontology generator uses `max_tokens=16000` specifically because Qwen3-series reasoning models consume thousands of tokens in internal reasoning before writing content — smaller token budgets result in empty responses.

**5. FTS5 trigram for CJK search**
Standard FTS5 tokenizers split on whitespace/punctuation, which fails for Chinese (no word boundaries). The trigram tokenizer creates n-grams of every 3 characters, enabling substring match on Chinese entity names without a dedicated CJK tokenizer.

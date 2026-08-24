"""Knowledge Graph Builder — builds graph from ContentList.

Inspired by RAG-Anything / LightRAG's multi-modal knowledge graph approach:

1. Structural entities — every content item becomes a graph node
2. LLM semantic extraction — entities + relationships from text chunks
3. Entity deduplication — merge same-named entities across chunks
4. Cross-modal relationship mapping
5. Hierarchical structure preservation ("belongs_to" chains)
6. Weighted relationship scoring
"""

import asyncio
import hashlib
import json as _json
import re
from typing import Callable, Optional

from agentic_rag.services.knowledge.content_list import ContentItem, ContentList, ContentType
from agentic_rag.services.knowledge.graph.index import Entity, KnowledgeGraph, Relation
from agentic_rag.config.prompts import Prompts


def _stable_id(name: str, prefix: str = "ent") -> str:
    """Generate a stable, deterministic entity ID from its name."""
    h = hashlib.md5(name.strip().lower().encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{h}"


def _normalize_name(name: str) -> str:
    """Normalize entity name for deduplication."""
    return name.strip().strip("'\".,;:!?()[]{}<>《》").strip()


class GraphBuilder:
    """Builds a knowledge graph from a ContentList.

    Two-layer construction:
    - **Structural layer**: every content item → entity, with belongs_to/nearby/describes edges
    - **Semantic layer** (optional, requires LLM): extracts fine-grained entities and
      semantic relationships from text chunks, deduplicates by name, and links to
      structural entities via "appears_in" edges.
    """

    def __init__(
        self,
        llm_func: Optional[Callable] = None,
        extract_entities: bool = True,
    ):
        """
        Args:
            llm_func: LLM function (text) -> str for entity/relation extraction.
            extract_entities: Whether to perform semantic entity extraction from text.
        """
        self.llm_func = llm_func
        self.extract_entities = extract_entities

    # ═══════════════════════════════════════════════════════════
    # Main entry point
    # ═══════════════════════════════════════════════════════════

    async def build(self, content_list: ContentList, doc_id: str = "",
                    existing_graph: KnowledgeGraph | None = None) -> KnowledgeGraph:
        """Build (or extend) a knowledge graph from a ContentList.

        Args:
            content_list: Parsed and processed document content.
            doc_id: Document identifier (used as namespace).
            existing_graph: If provided, extend this graph instead of creating new.

        Returns:
            Populated KnowledgeGraph.
        """
        graph = existing_graph if existing_graph is not None else KnowledgeGraph()
        import sys

        # ── Layer 1: Structural entities ──────────────────────
        structural_eids = []
        for i, item in enumerate(content_list.items):
            entity = await self._item_to_entity(item, i)
            eid = graph.add_entity(entity)
            structural_eids.append((eid, item))

        # ── Document entity ───────────────────────────────────
        doc_name = content_list.source or "Document"
        if hasattr(doc_name, "split"):
            doc_name = doc_name.split("/")[-1]  # filename only
        doc_entity = Entity(
            entity_id=_stable_id(doc_name, "doc"),
            name=doc_name,
            entity_type="document",
            properties={"doc_id": doc_id},
        )
        doc_eid = graph.add_entity(doc_entity)

        for eid, _ in structural_eids:
            graph.add_relation(eid, doc_eid, "belongs_to", weight=1.0)

        # ── Layer 2: Page-level adjacency ─────────────────────
        page_entities: dict[int, list[str]] = {}
        for eid, item in structural_eids:
            page_entities.setdefault(item.page_idx, []).append(eid)

        for page, eids in page_entities.items():
            for i in range(len(eids) - 1):
                graph.add_relation(eids[i], eids[i + 1], "nearby", weight=0.8)

        # ── Layer 3: Cross-modal references ───────────────────
        for page, eids in page_entities.items():
            page_ents = [graph.get_entity(eid) for eid in eids]
            page_ents = [e for e in page_ents if e is not None]
            text_ents = [e for e in page_ents if e.type == "text_chunk"]
            visual_ents = [e for e in page_ents if e.type in ("image", "table", "equation")]

            for text_e in text_ents:
                for visual_e in visual_ents:
                    graph.add_relation(
                        visual_e.id, text_e.id, "describes",
                        weight=0.6,
                        metadata={"page_idx": page},
                    )

        # ── Layer 4: Semantic entity extraction (LLM) ────────
        if self.extract_entities and self.llm_func:
            all_semantic_entities: dict[str, str] = {}  # normalized_name -> eid
            text_chunks = [
                (eid, item)
                for eid, item in structural_eids
                if item.type == ContentType.TEXT and item.text and len(item.text.strip()) > 30
            ]

            for eid, item in text_chunks:
                try:
                    chunk_entities, chunk_relations = await self._extract_from_chunk(
                        item.text
                    )
                    # Register semantic entities + "appears_in" links
                    for ent in chunk_entities:
                        norm = _normalize_name(ent["name"])
                        if norm in all_semantic_entities:
                            sem_eid = all_semantic_entities[norm]
                        else:
                            sem_entity = Entity(
                                entity_id=_stable_id(norm, "ent"),
                                name=ent["name"],
                                entity_type=ent.get("type", "CONCEPT").lower(),
                                properties={
                                    "description": ent.get("description", ""),
                                    "source_doc": doc_name,
                                },
                            )
                            sem_eid = graph.add_entity(sem_entity)
                            all_semantic_entities[norm] = sem_eid

                        # Link semantic entity → structural chunk
                        graph.add_relation(
                            sem_eid, eid, "appears_in",
                            weight=0.9,
                            metadata={"chunk_type": "text"},
                        )

                    # Register semantic relationships
                    for rel in chunk_relations:
                        src_norm = _normalize_name(rel["source"])
                        tgt_norm = _normalize_name(rel["target"])
                        if src_norm in all_semantic_entities and tgt_norm in all_semantic_entities:
                            src_eid = all_semantic_entities[src_norm]
                            tgt_eid = all_semantic_entities[tgt_norm]
                            if src_eid != tgt_eid:  # skip self-loops
                                graph.add_relation(
                                    src_eid, tgt_eid,
                                    rel.get("keywords", "related_to").split(",")[0].strip(),
                                    weight=0.85,
                                    metadata={
                                        "keywords": rel.get("keywords", ""),
                                        "description": rel.get("description", ""),
                                    },
                                )

                except Exception as e:
                    print(f"  [GraphBuilder] ⚠ Semantic extraction failed for chunk: {e}",
                          flush=True)
                    sys.stdout.flush()

            print(f"  [GraphBuilder] Semantic: {len(all_semantic_entities)} unique entities "
                  f"from {len(text_chunks)} chunks", flush=True)
            sys.stdout.flush()

        return graph

    # ═══════════════════════════════════════════════════════════
    # Helpers
    # ═══════════════════════════════════════════════════════════

    async def _extract_from_chunk(self, text: str) -> tuple[list[dict], list[dict]]:
        """Extract semantic entities and relationships from a text chunk via LLM.

        Returns:
            (entities_list, relationships_list) — each entity dict has
            name/type/description; each relation has source/target/keywords/description.
        """
        import sys
        prompt = Prompts.KG_ENTITY_EXTRACTION.format(text=text[:2000])
        try:
            raw = await self.llm_func(prompt)
            # Extract result from callable (may return str or object)
            if hasattr(raw, "content"):
                raw = raw.content
            elif hasattr(raw, "choices") and raw.choices:
                raw = raw.choices[0].message.content
            raw = str(raw).strip() if raw else ""
        except Exception as e:
            print(f"  [GraphBuilder] ⚠ LLM call failed: {e}", flush=True)
            sys.stdout.flush()
            return [], []

        if not raw:
            print(f"  [GraphBuilder] ⚠ LLM returned empty response", flush=True)
            sys.stdout.flush()
            return [], []

        # Strip Qwen <think>...</think> reasoning blocks (if present)
        raw = re.sub(r'<think>[\s\S]*?</think>', '', raw).strip()
        if not raw:
            print(f"  [GraphBuilder] ⚠ LLM response was all <think> block", flush=True)
            sys.stdout.flush()
            return [], []

        # Parse JSON from LLM output.
        # Qwen reasoning models prepend a "thinking process" before the actual
        # JSON — we must skip that and find the real output block.
        json_str = raw

        # Strategy 1: Find ```json fenced block (most reliable)
        fence_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', raw)
        if fence_match:
            json_str = fence_match.group(1).strip()
        else:
            # Strategy 2: Find the outermost JSON object that contains
            # "entities" key. Reasoning models put thinking text BEFORE the
            # JSON, so we scan from each `{` and pick the first one that
            # produces a complete, parseable dict with the right keys.
            json_str = ""
            for m in re.finditer(r'\{', raw):
                start = m.start()
                candidate = raw[start:]
                depth = 0
                end_pos = -1
                for i, ch in enumerate(candidate):
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            end_pos = i + 1
                            break
                if end_pos > 0:
                    candidate = candidate[:end_pos].strip()
                    # Quick check: does it look like our target JSON?
                    if '"entities"' in candidate or '"entity"' in candidate:
                        try:
                            _json.loads(candidate)
                            json_str = candidate
                            break  # found it
                        except _json.JSONDecodeError:
                            continue  # try next `{`

        # Strategy 3: Look for known JSON keys as a fallback
        if not json_str or '{' not in json_str:
            for marker in ('"entities"', '"entity"'):
                idx = raw.find(marker)
                if idx > 0:
                    brace_idx = raw.rfind('{', 0, idx)
                    if brace_idx >= 0:
                        json_str = raw[brace_idx:]
                        depth = 0
                        end_pos = -1
                        for i, ch in enumerate(json_str):
                            if ch == '{': depth += 1
                            elif ch == '}':
                                depth -= 1
                                if depth == 0:
                                    end_pos = i + 1
                                    break
                        if end_pos > 0:
                            json_str = json_str[:end_pos].strip()
                        break

        try:
            data = _json.loads(json_str)
        except _json.JSONDecodeError:
            # Attempt repair: try to fix common LLM JSON mistakes
            try:
                import json as _json_module
                data = _json.loads(json_str.replace("'", '"'))
            except Exception:
                print(f"  [GraphBuilder] ⚠ JSON parse failed, raw={raw[:120]}", flush=True)
                sys.stdout.flush()
                return [], []

        if not isinstance(data, dict):
            print(f"  [GraphBuilder] ⚠ Unexpected LLM output type: {type(data).__name__}", flush=True)
            sys.stdout.flush()
            return [], []

        entities = data.get("entities", []) or data.get("entity", []) or []
        relationships = data.get("relationships", []) or data.get("relations", []) or data.get("relationship", []) or []

        # Validate
        valid_entities = []
        for e in entities:
            if isinstance(e, dict) and e.get("name", "").strip():
                valid_entities.append({
                    "name": _normalize_name(e["name"]),
                    "type": str(e.get("type", "CONCEPT")).upper(),
                    "description": str(e.get("description", ""))[:200],
                })

        valid_relations = []
        for r in relationships:
            if isinstance(r, dict) and r.get("source", "").strip() and r.get("target", "").strip():
                valid_relations.append({
                    "source": _normalize_name(r["source"]),
                    "target": _normalize_name(r["target"]),
                    "keywords": str(r.get("keywords", "related_to"))[:100],
                    "description": str(r.get("description", ""))[:200],
                })

        if not valid_entities:
            print(f"  [GraphBuilder] ⚠ No valid entities extracted from chunk "
                  f"(text={text[:60]}...)", flush=True)
            sys.stdout.flush()

        return valid_entities, valid_relations

    async def _item_to_entity(self, item: ContentItem, index: int) -> Entity:
        """Convert a ContentItem to a structural graph Entity."""
        type_map = {
            ContentType.TEXT: "text_chunk",
            ContentType.IMAGE: "image",
            ContentType.TABLE: "table",
            ContentType.EQUATION: "equation",
            ContentType.VIDEO: "video_frame",
            ContentType.AUDIO: "audio_segment",
            ContentType.CODE: "code_block",
        }

        entity_type = type_map.get(item.type, "unknown")
        searchable = item.to_searchable_text()

        # Entity name: extract a meaningful title from the chunk text.
        # Strategy (in order of preference):
        #   1. Section header:  【...】 or \n...\n
        #   2. First complete sentence (ends with 。！？)
        #   3. First 50 chars, trimmed at word boundary
        if entity_type == "text_chunk" and searchable:
            text = searchable.strip()
            import re as _re
            # Try section header pattern: 【XXX】 or 第X页
            sec = _re.search(r'【(.+?)】', text)
            if sec:
                name = sec.group(0)  # e.g. "【药品名称】"
            else:
                # Try first line if it's a short heading
                first_line = text.split('\n')[0].strip()
                if len(first_line) <= 30 and not first_line.endswith(('。', '！', '？')):
                    name = first_line
                else:
                    # Find first complete sentence
                    name = text[:80]
                    for sep in ("。", "！", "？", "；"):
                        idx = name.find(sep)
                        if 15 < idx < 70:
                            name = name[:idx + 1]
                            break
                    # If name is still too long, trim at 50 chars
                    if len(name) > 50:
                        name = name[:50] + "…"
        elif searchable:
            name = searchable[:50]
        else:
            name = f"{entity_type}_{index}"

        # Generate stable ID from content hash
        id_src = (item.text or item.img_path or item.video_path or item.audio_path or str(index))
        eid = _stable_id(id_src[:200], entity_type[:6])

        return Entity(
            entity_id=eid,
            name=name,
            entity_type=entity_type,
            content_item=item,
            properties={
                "page_idx": item.page_idx,
                "content_type": item.type.value,
                "has_caption": bool(item.img_caption or item.table_caption or item.video_caption),
                "index": index,
            },
        )

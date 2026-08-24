"""Knowledge Graph Index — entity extraction, relationship mapping, and graph storage.

Inspired by RAG-Anything / LightRAG:
- Multi-modal entities from text + images + tables + equations
- Cross-modal relationship inference
- Hierarchical structure preservation ("belongs_to" chains)
- Weighted relationship scoring
"""

import asyncio
import hashlib
import uuid
from typing import Any, Callable, Optional

from agentic_rag.services.knowledge.content_list import ContentItem, ContentList, ContentType


class Entity:
    """A node in the knowledge graph."""

    def __init__(self, entity_id: str, name: str, entity_type: str,
                 content_item: Optional[ContentItem] = None, properties: dict = None):
        self.id = entity_id or uuid.uuid4().hex
        self.name = name
        self.type = entity_type  # "text_chunk", "image", "table", "equation", "concept"
        self.content_item = content_item
        self.properties = properties or {}
        self.embedding: Optional[list[float]] = None

    def __repr__(self):
        return f"Entity({self.name[:30]}, type={self.type})"


class Relation:
    """An edge in the knowledge graph."""

    def __init__(self, source_id: str, target_id: str, relation_type: str,
                 weight: float = 1.0, metadata: dict = None):
        self.source_id = source_id
        self.target_id = target_id
        self.type = relation_type  # "belongs_to", "references", "nearby", "describes"
        self.weight = weight
        self.metadata = metadata or {}

    def __repr__(self):
        return f"Relation({self.type}: {self.source_id[:8]} -> {self.target_id[:8]})"


class KnowledgeGraph:
    """In-memory knowledge graph with entities and relations.

    For production, this would be backed by a graph database (Neo4j, ArangoDB)
    or LightRAG's graph storage.
    """

    def __init__(self):
        self._entities: dict[str, Entity] = {}
        self._relations: list[Relation] = []
        # Indexes
        self._by_type: dict[str, list[str]] = {}   # type -> [entity_ids]
        self._by_page: dict[int, list[str]] = {}   # page_idx -> [entity_ids]
        self._adjacency: dict[str, list[Relation]] = {}  # entity_id -> [relations]

    # ── Entity CRUD ──────────────────────────────

    def add_entity(self, entity: Entity) -> str:
        self._entities[entity.id] = entity
        self._by_type.setdefault(entity.type, []).append(entity.id)
        if entity.content_item:
            page = entity.content_item.page_idx
            self._by_page.setdefault(page, []).append(entity.id)
        return entity.id

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        return self._entities.get(entity_id)

    def get_entities_by_type(self, entity_type: str) -> list[Entity]:
        ids = self._by_type.get(entity_type, [])
        return [self._entities[eid] for eid in ids if eid in self._entities]

    # ── Relation CRUD ────────────────────────────

    def add_relation(self, source_id: str, target_id: str,
                     relation_type: str = "references", weight: float = 1.0,
                     metadata: dict = None) -> Relation | None:
        # Deduplicate: skip if an identical (source, target, type) edge exists
        for existing in self._adjacency.get(source_id, []):
            if (existing.target_id == target_id
                    and existing.source_id == source_id
                    and existing.type == relation_type):
                return None  # already exists, skip
        rel = Relation(source_id, target_id, relation_type, weight, metadata)
        self._relations.append(rel)
        self._adjacency.setdefault(source_id, []).append(rel)
        # Inverse edge for bidirectional traversal
        inv_rel = Relation(
            target_id, source_id, f"inverse_{relation_type}", weight, metadata)
        self._adjacency.setdefault(target_id, []).append(inv_rel)
        return rel

    def get_neighbors(self, entity_id: str, relation_type: str = "") -> list[Entity]:
        """Get neighboring entities, optionally filtered by relation type."""
        rels = self._adjacency.get(entity_id, [])
        if relation_type:
            rels = [r for r in rels if r.type == relation_type]
        neighbors = []
        for rel in rels:
            neighbor_id = rel.target_id if rel.source_id == entity_id else rel.source_id
            if neighbor_id in self._entities:
                neighbors.append(self._entities[neighbor_id])
        return neighbors

    def traverse(self, entity_id: str, max_depth: int = 2,
                 relation_types: list[str] = None) -> list[Entity]:
        """BFS traversal from an entity."""
        visited = set()
        queue = [(entity_id, 0)]
        result = []
        while queue:
            eid, depth = queue.pop(0)
            if eid in visited or depth > max_depth:
                continue
            visited.add(eid)
            if eid in self._entities:
                result.append(self._entities[eid])
            for rel in self._adjacency.get(eid, []):
                neighbor = rel.target_id if rel.source_id == eid else rel.source_id
                if (relation_types is None or rel.type in relation_types):
                    queue.append((neighbor, depth + 1))
        return result

    # ── Properties ───────────────────────────────

    @property
    def entity_count(self) -> int:
        return len(self._entities)

    @property
    def relation_count(self) -> int:
        return len(self._relations)

    def stats(self) -> dict:
        return {
            "entities": self.entity_count,
            "relations": self.relation_count,
            "by_type": {t: len(ids) for t, ids in self._by_type.items()},
            "by_page": {p: len(ids) for p, ids in sorted(self._by_page.items())},
        }

    # ── Serialization ────────────────────────────

    def to_dict(self) -> dict:
        """Serialize graph for storage."""
        return {
            "entities": {
                eid: {
                    "id": e.id, "name": e.name, "type": e.type,
                    "properties": e.properties,
                    "content_type": e.content_item.type.value if e.content_item else None,
                    "content_text": e.content_item.to_searchable_text()[:500] if e.content_item else "",
                    # Persist enough ContentItem fields to reconstruct graph edges
                    "_item_text": e.content_item.text if e.content_item else "",
                    "_item_page_idx": e.content_item.page_idx if e.content_item else 0,
                }
                for eid, e in self._entities.items()
            },
            "relations": [
                {"source": r.source_id, "target": r.target_id,
                 "type": r.type, "weight": r.weight}
                for r in self._relations
            ],
        }

    def save_json(self, path: str) -> None:
        """Persist graph to a JSON file."""
        import json
        data = self.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "KnowledgeGraph":
        """Restore graph from a JSON file. Returns empty graph if file missing."""
        import json
        from pathlib import Path
        graph = cls()
        if not Path(path).exists():
            return graph
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return graph

        # Rebuild entities as lightweight stubs (no ContentItem — only used
        # for graph-traversal retrieval, not for full-text reconstruction).
        from agentic_rag.services.knowledge.content_list import ContentItem, ContentType
        for eid, ent in data.get("entities", {}).items():
            ctype_str = ent.get("content_type", "text")
            try:
                ctype = ContentType(ctype_str)
            except ValueError:
                ctype = ContentType.TEXT
            item = ContentItem(
                type=ctype,
                text=ent.get("_item_text", "")[:500],
                page_idx=ent.get("_item_page_idx", 0),
            )
            entity = Entity(
                entity_id=ent["id"],
                name=ent["name"],
                entity_type=ent["type"],
                content_item=item,
                properties=ent.get("properties", {}),
            )
            graph._entities[entity.id] = entity
            graph._by_type.setdefault(entity.type, []).append(entity.id)
            if entity.content_item:
                page = entity.content_item.page_idx
                graph._by_page.setdefault(page, []).append(entity.id)

        for rel in data.get("relations", []):
            graph.add_relation(
                rel["source"], rel["target"], rel["type"],
                weight=rel.get("weight", 1.0),
            )

        return graph

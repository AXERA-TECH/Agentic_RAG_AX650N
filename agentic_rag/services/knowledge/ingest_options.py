"""Per-request ingestion settings.

The knowledge pipeline is a process-wide singleton shared by every chat and
upload request.  Ingestion options therefore travel with the call instead of
being written onto the pipeline instance, which would make one request's
chunking or multimodal settings leak into a concurrent one.

``KnowledgePipeline`` constructor arguments still define the process defaults;
:class:`IngestOptions` overrides them for a single ``ingest*`` call.
"""

from __future__ import annotations

from dataclasses import dataclass

_INGEST_MODES = ("text", "multimodal")
_MM_METHODS = ("pure", "caption", "both")


@dataclass(frozen=True)
class IngestOptions:
    """Immutable ingestion configuration for one document."""

    ingest_mode: str = "multimodal"
    mm_method: str = "caption"
    enable_kg: bool = False
    extract_entities: bool | None = None
    chunk_size: int = 800
    chunk_overlap: int = 150

    def __post_init__(self) -> None:
        if self.ingest_mode not in _INGEST_MODES:
            raise ValueError(
                f"ingest_mode must be one of {_INGEST_MODES}, got {self.ingest_mode!r}"
            )
        if self.mm_method not in _MM_METHODS:
            raise ValueError(
                f"mm_method must be one of {_MM_METHODS}, got {self.mm_method!r}"
            )
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be greater than 0")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "chunk_overlap must be non-negative and smaller than chunk_size"
            )

    @property
    def should_extract_entities(self) -> bool:
        """Semantic extraction follows the KG toggle unless set explicitly."""
        if self.extract_entities is None:
            return self.enable_kg
        return bool(self.extract_entities) and self.enable_kg

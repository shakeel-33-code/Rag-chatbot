from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DocumentChunk:
    id: str
    content: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RetrievedChunk:
    id: str | None
    content: str
    score: float | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    fusion_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

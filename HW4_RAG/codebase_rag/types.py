from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceFile:
    abs_path: Path
    rel_path: str
    language: str
    content: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    file_path: str
    language: str
    start_line: int
    end_line: int
    content: str
    chunk_type: str = "text_chunk"
    symbol_name: str | None = None
    source: str = "codebase"


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class IndexStats:
    repo_id: str
    file_count: int
    chunk_count: int

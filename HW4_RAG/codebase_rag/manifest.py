from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from codebase_rag.repository import make_repo_id


@dataclass(frozen=True)
class IndexManifest:
    repo_id: str
    repo_path: str
    indexed_at: str
    file_count: int
    chunk_count: int
    embedding_model: str


class IndexManifestStore:
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.manifest_dir = base_path / "manifests"

    def manifest_path(self, repo_path: Path) -> Path:
        return self.manifest_dir / f"{make_repo_id(repo_path)}.json"

    def write(
        self,
        *,
        repo_path: Path,
        repo_id: str,
        file_count: int,
        chunk_count: int,
        embedding_model: str,
    ) -> IndexManifest:
        manifest = IndexManifest(
            repo_id=repo_id,
            repo_path=str(repo_path.expanduser().resolve()),
            indexed_at=datetime.now(UTC).isoformat(),
            file_count=file_count,
            chunk_count=chunk_count,
            embedding_model=embedding_model,
        )
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path(repo_path).write_text(
            json.dumps(asdict(manifest), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return manifest

    def read(self, repo_path: Path) -> IndexManifest | None:
        path = self.manifest_path(repo_path)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return IndexManifest(**payload)

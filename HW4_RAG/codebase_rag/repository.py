from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from codebase_rag.config import Settings


def make_repo_id(repo_path: Path) -> str:
    resolved = repo_path.expanduser().resolve()
    digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:10]
    safe_name = resolved.name.replace(" ", "_").replace("/", "_")
    return f"{safe_name}-{digest}"


def build_repo_tree(repo_path: Path, settings: Settings) -> dict[str, Any]:
    repo_path = repo_path.expanduser().resolve()
    if not repo_path.is_dir():
        raise ValueError(f"Repository path is not a directory: {repo_path}")

    def build_directory(path: Path) -> dict[str, Any]:
        children: list[dict[str, Any]] = []
        for child in sorted(path.iterdir(), key=lambda item: (item.is_file(), item.name.casefold())):
            if child.is_symlink() or child.name in settings.excluded_dir_names:
                continue
            if child.is_dir():
                node = build_directory(child)
                if node["children"]:
                    children.append(node)
                continue
            if child.name in settings.excluded_file_names:
                continue
            if any(child.name.endswith(suffix) for suffix in settings.excluded_file_suffixes):
                continue
            if child.suffix.lower() not in settings.supported_extensions:
                continue
            children.append(
                {
                    "name": child.name,
                    "type": "file",
                    "path": child.relative_to(repo_path).as_posix(),
                }
            )
        return {
            "name": path.name,
            "type": "directory",
            "path": "." if path == repo_path else path.relative_to(repo_path).as_posix(),
            "children": children,
        }

    return build_directory(repo_path)

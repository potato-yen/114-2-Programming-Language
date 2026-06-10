from __future__ import annotations

from pathlib import Path

from codebase_rag.config import Settings
from codebase_rag.language import detect_language
from codebase_rag.types import SourceFile


def _is_excluded(path: Path, repo_path: Path, settings: Settings) -> bool:
    rel_parts = path.relative_to(repo_path).parts
    return any(part in settings.excluded_dir_names for part in rel_parts)


def _read_text_file(path: Path) -> str | None:
    for encoding in ("utf-8", "utf-8-sig"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return None


def scan_repository(repo_path: Path, settings: Settings) -> list[SourceFile]:
    repo_path = repo_path.resolve()
    source_files: list[SourceFile] = []

    for path in sorted(repo_path.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        if _is_excluded(path, repo_path, settings):
            continue
        if path.suffix.lower() not in settings.supported_extensions:
            continue
        if path.name in settings.excluded_file_names:
            continue
        if any(path.name.endswith(suffix) for suffix in settings.excluded_file_suffixes):
            continue
        if path.stat().st_size > settings.max_file_bytes:
            continue

        language = detect_language(path)
        if language is None:
            continue

        content = _read_text_file(path)
        if content is None:
            continue

        source_files.append(
            SourceFile(
                abs_path=path,
                rel_path=path.relative_to(repo_path).as_posix(),
                language=language,
                content=content,
            )
        )

    return source_files

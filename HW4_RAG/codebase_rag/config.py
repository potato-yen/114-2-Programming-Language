from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str | None
    openrouter_model: str
    embedding_model: str
    vector_db_path: Path
    top_k: int
    max_file_bytes: int = 1_000_000
    chunk_size: int = 80
    chunk_overlap: int = 10
    supported_extensions: tuple[str, ...] = field(
        default=(
            ".py",
            ".js",
            ".ts",
            ".jsx",
            ".tsx",
            ".java",
            ".cpp",
            ".c",
            ".h",
            ".hpp",
            ".md",
            ".json",
            ".yaml",
            ".yml",
        )
    )
    excluded_dir_names: tuple[str, ...] = field(
        default=(
            ".git",
            "node_modules",
            "venv",
            ".venv",
            "__pycache__",
            "dist",
            "build",
            ".next",
            ".cache",
            ".idea",
            ".vscode",
        )
    )
    excluded_file_suffixes: tuple[str, ...] = field(default=(".min.js", ".lock"))
    excluded_file_names: tuple[str, ...] = field(
        default=(
            "package-lock.json",
            "yarn.lock",
            "pnpm-lock.yaml",
            "bun.lockb",
        )
    )

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
            openrouter_model=os.getenv(
                "OPENROUTER_MODEL",
                "openai/gpt-oss-120b:free",
            ),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL",
                "intfloat/multilingual-e5-small",
            ),
            vector_db_path=Path(os.getenv("VECTOR_DB_PATH", "./vector_db")),
            top_k=int(os.getenv("TOP_K", "6")),
        )

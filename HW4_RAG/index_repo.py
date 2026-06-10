from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from codebase_rag.config import Settings
from codebase_rag.indexer import build_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a codebase RAG index")
    parser.add_argument("repo_path", type=Path)
    parser.add_argument("--reset", action="store_true")
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()
    settings = Settings.from_env()
    try:
        stats = build_index(args.repo_path, settings=settings, reset=args.reset)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Indexed repo={stats.repo_id} files={stats.file_count} chunks={stats.chunk_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

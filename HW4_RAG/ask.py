from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from codebase_rag.config import Settings
from codebase_rag.llm import OpenRouterClient
from codebase_rag.prompt import build_answer_prompt
from codebase_rag.retriever import (
    build_abstain_message,
    format_retrieval_only_answer,
    render_retrieved_context,
    search_codebase,
    should_abstain,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask the indexed codebase a question")
    parser.add_argument("question")
    parser.add_argument("--repo-path", type=Path, default=Path("./sample_project"))
    parser.add_argument("--no-llm", action="store_true")
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()
    settings = Settings.from_env()
    try:
        hits = search_codebase(args.repo_path, args.question, settings)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if should_abstain(args.question, hits):
        print(build_abstain_message(args.question))
        return 0

    if args.no_llm or not settings.openrouter_api_key:
        print(format_retrieval_only_answer(args.question, hits))
        return 0

    try:
        prompt = build_answer_prompt(args.question, render_retrieved_context(hits))
        client = OpenRouterClient(
            api_key=settings.openrouter_api_key,
            model=settings.openrouter_model,
        )
        print(client.answer(prompt))
    except Exception:
        print(format_retrieval_only_answer(args.question, hits))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

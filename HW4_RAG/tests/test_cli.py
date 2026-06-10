from __future__ import annotations

import argparse
from pathlib import Path

import ask
import index_repo


def test_index_cli_prints_embedding_runtime_error(monkeypatch, capsys):
    monkeypatch.setattr(
        index_repo,
        "parse_args",
        lambda: argparse.Namespace(repo_path=Path("./sample_project"), reset=True),
    )

    def broken_build_index(*args, **kwargs):
        raise RuntimeError("embedding download failed")

    monkeypatch.setattr(index_repo, "build_index", broken_build_index)

    exit_code = index_repo.main()

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "embedding download failed" in captured.err


def test_ask_cli_prints_search_runtime_error(monkeypatch, capsys):
    monkeypatch.setattr(
        ask,
        "parse_args",
        lambda: argparse.Namespace(
            question="登入功能在哪裡？",
            repo_path=Path("./sample_project"),
            no_llm=True,
        ),
    )

    def broken_search(*args, **kwargs):
        raise RuntimeError("embedding download failed")

    monkeypatch.setattr(ask, "search_codebase", broken_search)

    exit_code = ask.main()

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "embedding download failed" in captured.err

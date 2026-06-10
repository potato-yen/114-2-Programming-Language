from codebase_rag.config import Settings
from codebase_rag.indexer import build_index
from codebase_rag.repository import make_repo_id


class FakeEmbedder:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float(index), 0.0, 0.0] for index, _ in enumerate(texts, start=1)]


def test_build_index_scans_chunks_and_persists(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "src" / "auth.py").write_text(
        "def login():\n    return True\n",
        encoding="utf-8",
    )

    settings = Settings.from_env()
    stats = build_index(
        repo_path=repo,
        settings=settings,
        embedder=FakeEmbedder(),
        vector_db_path=tmp_path / "vector_db",
        reset=True,
    )

    assert stats.repo_id == make_repo_id(repo)
    assert stats.file_count == 1
    assert stats.chunk_count == 1
    assert (tmp_path / "vector_db" / f"{make_repo_id(repo)}__sparse.json").exists()


def test_build_index_wraps_embedding_load_errors(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "src" / "auth.py").write_text(
        "def login():\n    return True\n",
        encoding="utf-8",
    )

    class BrokenEmbedder:
        def __init__(self, model_name: str):
            raise OSError("network unavailable")

    monkeypatch.setattr("codebase_rag.indexer.SentenceTransformerEmbedder", BrokenEmbedder)

    settings = Settings.from_env()

    try:
        build_index(
            repo_path=repo,
            settings=settings,
            vector_db_path=tmp_path / "vector_db",
            reset=True,
        )
    except RuntimeError as exc:
        assert settings.embedding_model in str(exc)
        assert "download" in str(exc)
    else:
        raise AssertionError("Expected build_index() to raise RuntimeError")

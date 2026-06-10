from fastapi.testclient import TestClient

from codebase_rag.api import app
from codebase_rag.types import Chunk, IndexStats, SearchHit


def test_repo_status_returns_tree_and_missing_index(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "main.py").write_text("print('hi')", encoding="utf-8")
    monkeypatch.setenv("VECTOR_DB_PATH", str(tmp_path / "vector_db"))
    client = TestClient(app)

    response = client.post("/api/repo/status", json={"repo_path": str(repo)})

    assert response.status_code == 200
    payload = response.json()
    assert payload["index_available"] is False
    assert payload["tree"]["name"] == "repo"
    assert "main.py" in str(payload["tree"])


def test_repo_index_returns_stats(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setenv("VECTOR_DB_PATH", str(tmp_path / "vector_db"))
    monkeypatch.setattr(
        "codebase_rag.api.build_index",
        lambda **kwargs: IndexStats(repo_id="repo-123", file_count=4, chunk_count=9),
    )
    client = TestClient(app)

    response = client.post("/api/repo/index", json={"repo_path": str(repo)})

    assert response.status_code == 200
    assert response.json()["repo_id"] == "repo-123"
    assert response.json()["chunk_count"] == 9


def test_query_returns_grounded_sources(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    hit = SearchHit(
        chunk=Chunk(
            chunk_id="src/auth.py:1-8",
            file_path="src/auth.py",
            language="python",
            start_line=1,
            end_line=8,
            content="def login(): return True",
            symbol_name="login",
        ),
        score=0.91,
    )
    monkeypatch.setattr("codebase_rag.api.search_codebase", lambda *args, **kwargs: [hit])
    monkeypatch.setattr("codebase_rag.api.should_abstain", lambda *args, **kwargs: False)
    client = TestClient(app)

    response = client.post(
        "/api/query",
        json={"repo_path": str(repo), "question": "Where is login?", "use_llm": False},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["abstained"] is False
    assert payload["answer_mode"] == "retrieval"
    assert payload["answer_notice"] is None
    assert payload["sources"][0]["file_path"] == "src/auth.py"
    assert payload["sources"][0]["start_line"] == 1


def test_query_reports_llm_mode_when_llm_answers(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    hit = SearchHit(
        chunk=Chunk(
            chunk_id="src/auth.py:1-8",
            file_path="src/auth.py",
            language="python",
            start_line=1,
            end_line=8,
            content="def login(): return True",
            symbol_name="login",
        ),
        score=0.91,
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("codebase_rag.api.search_codebase", lambda *args, **kwargs: [hit])
    monkeypatch.setattr("codebase_rag.api.should_abstain", lambda *args, **kwargs: False)
    monkeypatch.setattr("codebase_rag.api.OpenRouterClient.answer", lambda *args, **kwargs: "LLM answer")
    client = TestClient(app)

    response = client.post(
        "/api/query",
        json={"repo_path": str(repo), "question": "Where is login?", "use_llm": True},
    )

    payload = response.json()
    assert payload["answer"] == "LLM answer"
    assert payload["answer_mode"] == "llm"
    assert payload["answer_notice"] is None


def test_query_reports_retrieval_fallback_when_llm_is_unavailable(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    hit = SearchHit(
        chunk=Chunk(
            chunk_id="src/auth.py:1-8",
            file_path="src/auth.py",
            language="python",
            start_line=1,
            end_line=8,
            content="def login(): return True",
            symbol_name="login",
        ),
        score=0.91,
    )
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("codebase_rag.api.search_codebase", lambda *args, **kwargs: [hit])
    monkeypatch.setattr("codebase_rag.api.should_abstain", lambda *args, **kwargs: False)
    client = TestClient(app)

    response = client.post(
        "/api/query",
        json={"repo_path": str(repo), "question": "Where is login?", "use_llm": True},
    )

    payload = response.json()
    assert payload["answer_mode"] == "retrieval"
    assert payload["answer_notice"] == "LLM answer requested, but OPENROUTER_API_KEY is not configured."


def test_file_endpoint_rejects_path_traversal(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    client = TestClient(app)

    response = client.post(
        "/api/file",
        json={"repo_path": str(repo), "file_path": "../secret.txt"},
    )

    assert response.status_code == 400

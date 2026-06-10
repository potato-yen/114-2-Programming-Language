from pathlib import Path

from codebase_rag.config import Settings


def test_settings_defaults(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("VECTOR_DB_PATH", raising=False)
    monkeypatch.delenv("TOP_K", raising=False)

    settings = Settings.from_env()

    assert settings.embedding_model == "intfloat/multilingual-e5-small"
    assert settings.openrouter_model == "openai/gpt-oss-120b:free"
    assert settings.vector_db_path == Path("./vector_db")
    assert settings.top_k == 6
    assert ".py" in settings.supported_extensions
    assert settings.openrouter_api_key is None


def test_settings_overrides(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "openrouter/test-model")
    monkeypatch.setenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    monkeypatch.setenv("VECTOR_DB_PATH", "./custom_db")
    monkeypatch.setenv("TOP_K", "4")

    settings = Settings.from_env()

    assert settings.openrouter_api_key == "test-key"
    assert settings.openrouter_model == "openrouter/test-model"
    assert settings.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.vector_db_path == Path("./custom_db")
    assert settings.top_k == 4

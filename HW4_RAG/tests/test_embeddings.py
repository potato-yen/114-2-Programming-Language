from codebase_rag.embeddings import SentenceTransformerEmbedder, _load_model


def test_embedder_uses_local_cache_before_network(monkeypatch):
    _load_model.cache_clear()
    calls: list[tuple[str, bool]] = []

    class FakeModel:
        def encode(self, text, normalize_embeddings=True):
            _ = text
            _ = normalize_embeddings
            return [0.1, 0.2, 0.3]

    class FakeSentenceTransformer:
        def __new__(cls, model_name: str, local_files_only: bool = False):
            calls.append((model_name, local_files_only))
            return FakeModel()

    monkeypatch.setattr(
        "codebase_rag.embeddings._sentence_transformer_cls",
        lambda: FakeSentenceTransformer,
    )

    embedder = SentenceTransformerEmbedder("intfloat/multilingual-e5-small")

    assert calls == [("intfloat/multilingual-e5-small", True)]
    assert embedder.embed_query("login") == [0.1, 0.2, 0.3]


def test_embedder_downloads_model_when_local_cache_is_missing(monkeypatch):
    _load_model.cache_clear()
    calls: list[tuple[str, bool]] = []

    class FakeModel:
        pass

    class FakeSentenceTransformer:
        def __new__(cls, model_name: str, local_files_only: bool = False):
            calls.append((model_name, local_files_only))
            if local_files_only:
                raise OSError("model is not cached")
            return FakeModel()

    monkeypatch.setattr(
        "codebase_rag.embeddings._sentence_transformer_cls",
        lambda: FakeSentenceTransformer,
    )

    SentenceTransformerEmbedder("new-model")

    assert calls == [
        ("new-model", True),
        ("new-model", False),
    ]


def test_embedder_reuses_loaded_model(monkeypatch):
    _load_model.cache_clear()
    calls: list[str] = []

    class FakeModel:
        def encode(self, text, normalize_embeddings=True):
            _ = text
            _ = normalize_embeddings
            return [0.1]

    class FakeSentenceTransformer:
        def __new__(cls, model_name: str, local_files_only: bool = False):
            _ = local_files_only
            calls.append(model_name)
            return FakeModel()

    monkeypatch.setattr(
        "codebase_rag.embeddings._sentence_transformer_cls",
        lambda: FakeSentenceTransformer,
    )

    first = SentenceTransformerEmbedder("cached-model")
    second = SentenceTransformerEmbedder("cached-model")

    assert first.model is second.model
    assert calls == ["cached-model"]

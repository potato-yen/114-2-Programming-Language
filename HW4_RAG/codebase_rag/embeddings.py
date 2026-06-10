from __future__ import annotations

from functools import lru_cache


def _sentence_transformer_cls():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer


def _to_list(vector) -> list[float]:
    if hasattr(vector, "tolist"):
        return vector.tolist()
    return list(vector)


@lru_cache(maxsize=4)
def _load_model(model_name: str):
    sentence_transformer = _sentence_transformer_cls()
    try:
        return sentence_transformer(model_name, local_files_only=True)
    except Exception:
        return sentence_transformer(model_name)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        self.model = _load_model(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        prefixed = [f"passage: {text}" for text in texts]
        vectors = self.model.encode(prefixed, normalize_embeddings=True)
        return [_to_list(vector) for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode(f"query: {text}", normalize_embeddings=True)
        return _to_list(vector)


def build_embedding_error(model_name: str, exc: Exception) -> RuntimeError:
    _ = exc
    return RuntimeError(
        f"Failed to load or run embedding model '{model_name}'. "
        "Ensure network access for the first download or set EMBEDDING_MODEL "
        "to a cached/local model."
    )

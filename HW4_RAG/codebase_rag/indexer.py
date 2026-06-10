from __future__ import annotations

from pathlib import Path
from typing import Protocol

from codebase_rag.chunker import chunk_source_file, render_chunk_document
from codebase_rag.config import Settings
from codebase_rag.embeddings import SentenceTransformerEmbedder, build_embedding_error
from codebase_rag.manifest import IndexManifestStore
from codebase_rag.repository import make_repo_id
from codebase_rag.scanner import scan_repository
from codebase_rag.sparse_index import SparseIndexStore
from codebase_rag.types import IndexStats
from codebase_rag.vectorstore import ChromaVectorStore


class TextEmbedder(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


def build_index(
    repo_path: Path,
    settings: Settings,
    embedder: TextEmbedder | None = None,
    vector_db_path: Path | None = None,
    reset: bool = False,
) -> IndexStats:
    repo_path = repo_path.resolve()
    repo_id = make_repo_id(repo_path)
    source_files = scan_repository(repo_path, settings)
    chunks = [
        chunk
        for source_file in source_files
        for chunk in chunk_source_file(source_file, settings)
    ]

    vectorstore = ChromaVectorStore(vector_db_path or settings.vector_db_path)
    sparse_index = SparseIndexStore(vector_db_path or settings.vector_db_path)
    if reset:
        vectorstore.reset(repo_id)
        sparse_index.reset(repo_id)

    if chunks:
        documents = [render_chunk_document(chunk) for chunk in chunks]
        if embedder is None:
            try:
                active_embedder = SentenceTransformerEmbedder(settings.embedding_model)
                embeddings = active_embedder.embed_texts(documents)
            except Exception as exc:
                raise build_embedding_error(settings.embedding_model, exc) from exc
        else:
            embeddings = embedder.embed_texts(documents)
        vectorstore.upsert(repo_id, chunks, embeddings)
        sparse_index.upsert(repo_id, chunks)

    stats = IndexStats(
        repo_id=repo_id,
        file_count=len(source_files),
        chunk_count=len(chunks),
    )
    IndexManifestStore(vector_db_path or settings.vector_db_path).write(
        repo_path=repo_path,
        repo_id=repo_id,
        file_count=stats.file_count,
        chunk_count=stats.chunk_count,
        embedding_model=settings.embedding_model,
    )
    return stats

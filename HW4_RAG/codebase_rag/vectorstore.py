from __future__ import annotations

from pathlib import Path

import chromadb

from codebase_rag.types import Chunk, SearchHit


def _collection_name(repo_id: str) -> str:
    return repo_id.replace("/", "_").replace(" ", "_")


class ChromaVectorStore:
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(base_path))

    def reset(self, repo_id: str) -> None:
        name = _collection_name(repo_id)
        try:
            self.client.delete_collection(name)
        except Exception:
            pass
        self.client.get_or_create_collection(name=name, metadata={"repo_id": repo_id})

    def upsert(self, repo_id: str, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return

        collection = self.client.get_or_create_collection(
            name=_collection_name(repo_id),
            metadata={"repo_id": repo_id},
        )
        collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.content for chunk in chunks],
            embeddings=embeddings,
            metadatas=[
                {
                    "file_path": chunk.file_path,
                    "language": chunk.language,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "chunk_type": chunk.chunk_type,
                    "symbol_name": chunk.symbol_name,
                    "source": chunk.source,
                }
                for chunk in chunks
            ],
        )

    def query(self, repo_id: str, query_embedding: list[float], top_k: int) -> list[SearchHit]:
        collection = self.client.get_or_create_collection(
            name=_collection_name(repo_id),
            metadata={"repo_id": repo_id},
        )
        if collection.count() == 0:
            return []

        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        hits: list[SearchHit] = []
        for metadata, document, chunk_id, distance in zip(
            result["metadatas"][0],
            result["documents"][0],
            result["ids"][0],
            result["distances"][0],
            strict=False,
        ):
            hits.append(
                SearchHit(
                    chunk=Chunk(
                        chunk_id=chunk_id,
                        file_path=str(metadata["file_path"]),
                        language=str(metadata["language"]),
                        start_line=int(metadata["start_line"]),
                        end_line=int(metadata["end_line"]),
                        content=document,
                        chunk_type=str(metadata.get("chunk_type", "text_chunk")),
                        symbol_name=metadata.get("symbol_name"),
                        source=str(metadata.get("source", "codebase")),
                    ),
                    score=1 - float(distance),
                )
            )
        return hits

    def list_chunks(self, repo_id: str) -> list[Chunk]:
        collection = self.client.get_or_create_collection(
            name=_collection_name(repo_id),
            metadata={"repo_id": repo_id},
        )
        if collection.count() == 0:
            return []

        result = collection.get(include=["documents", "metadatas"])
        chunks: list[Chunk] = []
        for metadata, document, chunk_id in zip(
            result["metadatas"],
            result["documents"],
            result["ids"],
            strict=False,
        ):
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    file_path=str(metadata["file_path"]),
                    language=str(metadata["language"]),
                    start_line=int(metadata["start_line"]),
                    end_line=int(metadata["end_line"]),
                    content=document,
                    chunk_type=str(metadata.get("chunk_type", "text_chunk")),
                    symbol_name=metadata.get("symbol_name"),
                    source=str(metadata.get("source", "codebase")),
                )
            )
        return chunks

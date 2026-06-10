from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from codebase_rag.chunker import render_chunk_document
from codebase_rag.config import Settings
from codebase_rag.embeddings import SentenceTransformerEmbedder, build_embedding_error
from codebase_rag.repository import make_repo_id
from codebase_rag.sparse_index import SparseIndexStore, build_sparse_terms, search_chunks
from codebase_rag.types import Chunk, SearchHit
from codebase_rag.vectorstore import ChromaVectorStore


class QueryEmbedder(Protocol):
    def embed_query(self, text: str) -> list[float]: ...


@dataclass(frozen=True)
class FileHitGroup:
    file_path: str
    language: str
    primary_hit: SearchHit
    hits: list[SearchHit]
    line_ranges: list[tuple[int, int]]
    symbol_names: list[str]


LOCKFILE_NAMES = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml", "bun.lockb"}
CODE_LANGUAGES = {
    "python",
    "javascript",
    "javascriptreact",
    "typescript",
    "typescriptreact",
    "java",
    "cpp",
    "c",
}
GENERIC_SYMBOLS = {"onsubmit", "handlesubmit", "submit", "render"}


def _chunk_terms(chunk: Chunk) -> tuple[set[str], set[str], set[str]]:
    return (
        build_sparse_terms(chunk.file_path),
        build_sparse_terms(chunk.symbol_name or ""),
        build_sparse_terms(chunk.content),
    )


def _match_counts(query_terms: set[str], chunk: Chunk) -> tuple[int, int, int]:
    path_terms, symbol_terms, content_terms = _chunk_terms(chunk)
    return (
        len(query_terms & path_terms),
        len(query_terms & symbol_terms),
        len(query_terms & content_terms),
    )


def _dense_rank_score(rank: int, *, k: int = 60) -> float:
    return 1.0 / (k + rank)


def _is_low_signal_chunk(chunk: Chunk) -> bool:
    file_name = Path(chunk.file_path).name
    if file_name in LOCKFILE_NAMES:
        return True
    if chunk.language == "markdown":
        return True
    if chunk.file_path.startswith("docs/") or file_name == "README.md":
        return True
    return False


def sparse_search(question: str, chunks: list[Chunk], top_k: int) -> list[SearchHit]:
    filtered_chunks = [chunk for chunk in chunks if not _is_low_signal_chunk(chunk)]
    return search_chunks(question, filtered_chunks, top_k)


def fuse_hits(dense_hits: list[SearchHit], sparse_hits: list[SearchHit]) -> list[SearchHit]:
    chunk_by_id: dict[str, Chunk] = {}
    fused_scores: dict[str, float] = {}

    for rank, hit in enumerate(dense_hits, start=1):
        chunk_by_id[hit.chunk.chunk_id] = hit.chunk
        fused_scores[hit.chunk.chunk_id] = fused_scores.get(hit.chunk.chunk_id, 0.0) + _dense_rank_score(rank)

    for rank, hit in enumerate(sparse_hits, start=1):
        chunk_by_id[hit.chunk.chunk_id] = hit.chunk
        fused_scores[hit.chunk.chunk_id] = fused_scores.get(hit.chunk.chunk_id, 0.0) + _dense_rank_score(rank)

    if not fused_scores:
        return []

    max_score = max(fused_scores.values())
    hits = [
        SearchHit(chunk=chunk_by_id[chunk_id], score=score / max_score)
        for chunk_id, score in fused_scores.items()
    ]
    hits.sort(key=lambda hit: hit.score, reverse=True)
    return hits


def _score_hit(question: str, hit: SearchHit) -> float:
    return _score_hit_for_terms(build_sparse_terms(question), hit)


def _score_hit_for_terms(query_terms: set[str], hit: SearchHit) -> float:
    path = hit.chunk.file_path
    file_name = Path(path).name
    symbol_name = hit.chunk.symbol_name or ""
    score = hit.score
    path_overlap, symbol_overlap, content_overlap = _match_counts(query_terms, hit.chunk)

    if hit.chunk.language in CODE_LANGUAGES:
        score += 0.2
    if hit.chunk.language == "markdown":
        score -= 0.7
    if hit.chunk.language in {"json", "yaml"}:
        score -= 0.55

    if file_name in LOCKFILE_NAMES:
        score -= 1.5
    if path.startswith("docs/") or file_name == "README.md":
        score -= 0.5
    if "/error" in path or "/errors" in path or "messages" in file_name.casefold():
        score -= 0.45
    if "maperror" in file_name.casefold():
        score -= 0.2

    score += path_overlap * 0.25
    score += symbol_overlap * 0.35
    score += content_overlap * 0.10

    if symbol_name and symbol_name.casefold() in GENERIC_SYMBOLS and symbol_overlap == 0:
        score -= 0.2

    return score


def _score_file_group(query_terms: set[str], file_hits: list[SearchHit]) -> float:
    hit_scores = [_score_hit_for_terms(query_terms, hit) for hit in file_hits]
    matched_hits = 0
    total_path_overlap = 0
    total_symbol_overlap = 0
    total_content_overlap = 0
    matched_symbols: set[str] = set()
    matched_terms: set[str] = set()

    for hit in file_hits:
        path_terms, symbol_terms, content_terms = _chunk_terms(hit.chunk)
        path_overlap, symbol_overlap, content_overlap = _match_counts(query_terms, hit.chunk)
        if path_overlap or symbol_overlap or content_overlap:
            matched_hits += 1
            total_path_overlap += path_overlap
            total_symbol_overlap += symbol_overlap
            total_content_overlap += content_overlap
            matched_terms.update(query_terms & path_terms)
            matched_terms.update(query_terms & symbol_terms)
            matched_terms.update(query_terms & content_terms)
            if hit.chunk.symbol_name:
                matched_symbols.add(hit.chunk.symbol_name)

    score = max(hit_scores)
    if matched_terms:
        score += len(matched_terms) * 0.60
        score += (len(matched_terms) / len(query_terms)) * 1.00
    if matched_hits > 1:
        score += 0.08 * (matched_hits - 1)
    score += min(total_path_overlap, 3) * 0.06
    score += min(total_symbol_overlap, 3) * 0.10
    score += min(total_content_overlap, 6) * 0.04
    score += min(len(matched_symbols), 3) * 0.04
    return score


def rerank_hits(question: str, hits: list[SearchHit]) -> list[SearchHit]:
    query_terms = build_sparse_terms(question)
    grouped_hits = group_hits_by_file(hits)
    file_scores = {
        group.file_path: _score_file_group(query_terms, group.hits)
        for group in grouped_hits
    }

    reranked_groups = sorted(
        grouped_hits,
        key=lambda group: (
            file_scores[group.file_path],
            _score_hit_for_terms(query_terms, group.primary_hit),
            group.primary_hit.score,
        ),
        reverse=True,
    )

    reranked_hits: list[SearchHit] = []
    for group in reranked_groups:
        reranked_hits.extend(
            sorted(
                group.hits,
                key=lambda hit: (
                    _score_hit_for_terms(query_terms, hit),
                    hit.score,
                    -hit.chunk.start_line,
                ),
                reverse=True,
            )
        )
    return reranked_hits


def _merge_line_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not ranges:
        return []

    sorted_ranges = sorted(ranges)
    merged: list[tuple[int, int]] = [sorted_ranges[0]]
    for start, end in sorted_ranges[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _format_line_ranges(ranges: list[tuple[int, int]]) -> str:
    return ", ".join(
        f"{start}-{end}" if start != end else f"{start}"
        for start, end in ranges
    )


def group_hits_by_file(hits: list[SearchHit], max_files: int | None = None) -> list[FileHitGroup]:
    ordered_paths: list[str] = []
    hits_by_path: dict[str, list[SearchHit]] = {}

    for hit in hits:
        path = hit.chunk.file_path
        if path not in hits_by_path:
            ordered_paths.append(path)
            hits_by_path[path] = []
        hits_by_path[path].append(hit)

    groups: list[FileHitGroup] = []
    for path in ordered_paths:
        file_hits = hits_by_path[path]
        sorted_hits = sorted(file_hits, key=lambda hit: (-hit.score, hit.chunk.start_line))
        primary_hit = sorted_hits[0]
        line_ranges = _merge_line_ranges(
            [(hit.chunk.start_line, hit.chunk.end_line) for hit in file_hits]
        )
        symbol_names: list[str] = []
        seen_symbols: set[str] = set()
        for hit in file_hits:
            symbol_name = hit.chunk.symbol_name
            if not symbol_name or symbol_name in seen_symbols:
                continue
            seen_symbols.add(symbol_name)
            symbol_names.append(symbol_name)

        groups.append(
            FileHitGroup(
                file_path=path,
                language=primary_hit.chunk.language,
                primary_hit=primary_hit,
                hits=sorted(file_hits, key=lambda hit: hit.chunk.start_line),
                line_ranges=line_ranges,
                symbol_names=symbol_names,
            )
        )

    if max_files is not None:
        return groups[:max_files]
    return groups


def should_abstain(question: str, hits: list[SearchHit]) -> bool:
    if not hits:
        return True

    query_terms = build_sparse_terms(question)
    if not query_terms:
        return False

    top_hits = hits[:5]
    matched_hits = 0
    code_hits = 0
    for hit in top_hits:
        path_overlap, symbol_overlap, content_overlap = _match_counts(query_terms, hit.chunk)
        if path_overlap or symbol_overlap or content_overlap:
            matched_hits += 1
        if hit.chunk.language in CODE_LANGUAGES:
            code_hits += 1

    if matched_hits == 0:
        top_score = top_hits[0].score
        next_score = top_hits[1].score if len(top_hits) > 1 else 0.0
        if code_hits > 0 and top_score >= 0.9 and (top_score - next_score) >= 0.15:
            return False
        return True
    if code_hits == 0:
        return True
    return False


def build_abstain_message(question: str) -> str:
    return f"找不到足夠證據回答這個問題：{question}"


def search_codebase(
    repo_path: Path,
    question: str,
    settings: Settings,
    embedder: QueryEmbedder | None = None,
    vector_db_path: Path | None = None,
) -> list[SearchHit]:
    repo_id = make_repo_id(repo_path)
    vectorstore = ChromaVectorStore(vector_db_path or settings.vector_db_path)
    if embedder is None:
        try:
            active_embedder = SentenceTransformerEmbedder(settings.embedding_model)
            query_embedding = active_embedder.embed_query(question)
        except Exception as exc:
            raise build_embedding_error(settings.embedding_model, exc) from exc
    else:
        query_embedding = embedder.embed_query(question)
    dense_hits = [
        hit
        for hit in vectorstore.query(repo_id, query_embedding, max(settings.top_k * 40, 200))
        if not _is_low_signal_chunk(hit.chunk)
    ]
    sparse_store = SparseIndexStore(vector_db_path or settings.vector_db_path)
    sparse_hits = [
        hit
        for hit in sparse_store.search(repo_id, question, max(settings.top_k * 10, 50))
        if not _is_low_signal_chunk(hit.chunk)
    ]
    if not sparse_hits:
        sparse_hits = sparse_search(question, vectorstore.list_chunks(repo_id), max(settings.top_k * 10, 50))
    fused_hits = fuse_hits(dense_hits, sparse_hits)
    return rerank_hits(question, fused_hits)[: settings.top_k]


def format_retrieval_only_answer(question: str, hits: list[SearchHit]) -> str:
    if should_abstain(question, hits):
        return build_abstain_message(question)

    lines = ["可能相關位置：", ""]
    for index, group in enumerate(group_hits_by_file(hits), start=1):
        symbol = ""
        if len(group.symbol_names) == 1:
            symbol = f", {group.symbol_names[0]}()"
        lines.append(
            f"{index}. {group.file_path}{symbol}, lines {_format_line_ranges(group.line_ranges)}"
        )
        lines.append(
            f"   理由：檔案中的相關 chunk 與問題在檔名、symbol 或內容上有語意重疊，代表分數 {group.primary_hit.score:.2f}。"
        )
        lines.append("")
    return "\n".join(lines).strip()


def render_retrieved_context(hits: list[SearchHit]) -> str:
    sections: list[str] = []
    for group in group_hits_by_file(hits):
        chunk_bodies = "\n\n".join(hit.chunk.content for hit in group.hits)
        symbols = ", ".join(group.symbol_names) if group.symbol_names else "N/A"
        sections.append(
            (
                f"File: {group.file_path}\n"
                f"Language: {group.language}\n"
                f"Symbols: {symbols}\n"
                f"Lines: {_format_line_ranges(group.line_ranges)}\n\n"
                f"Code:\n{chunk_bodies}"
            )
        )
    return "\n\n".join(sections)

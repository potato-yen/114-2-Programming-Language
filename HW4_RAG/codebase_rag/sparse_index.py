from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from codebase_rag.types import Chunk, SearchHit

ASCII_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")
CJK_SPAN_RE = re.compile(r"[\u4e00-\u9fff]+")
CAMEL_PART_RE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")

FIELD_WEIGHTS = {
    "path": 2.5,
    "symbol": 3.0,
    "content": 1.0,
}
BM25_K1 = 1.5
BM25_B = 0.75
ENGLISH_QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "what",
    "where",
    "which",
    "with",
}


def _normalize(text: str) -> str:
    return text.casefold()


def _english_term_variants(term: str) -> list[str]:
    variants = [term]
    if len(term) <= 3:
        return variants
    if term.endswith("ies") and len(term) > 4:
        variants.append(f"{term[:-3]}y")
    elif term.endswith(("ches", "shes", "xes", "zes")):
        variants.append(term[:-2])
    elif term.endswith("s") and not term.endswith(("ss", "us", "is")):
        variants.append(term[:-1])
    return variants


def _split_identifier(token: str) -> list[str]:
    parts: list[str] = []
    for piece in re.split(r"[_\-./()]+", token):
        if not piece:
            continue
        normalized_piece = _normalize(piece)
        if len(normalized_piece) >= 2:
            parts.extend(_english_term_variants(normalized_piece))
        camel_parts = CAMEL_PART_RE.findall(piece)
        normalized_parts = [_normalize(part) for part in camel_parts if part]
        for part in normalized_parts:
            if len(part) >= 2:
                parts.extend(_english_term_variants(part))
        for index in range(len(normalized_parts) - 1):
            merged = "".join(normalized_parts[index : index + 2])
            if len(merged) >= 2:
                parts.append(merged)
    return parts


def _cjk_bigrams(text: str) -> list[str]:
    terms: list[str] = []
    for match in CJK_SPAN_RE.finditer(text):
        span = match.group(0)
        if len(span) <= 2:
            terms.append(span)
            continue
        terms.append(span)
        for index in range(len(span) - 1):
            terms.append(span[index : index + 2])
    return terms


def tokenize_sparse_text(text: str) -> list[str]:
    terms: list[str] = []
    for token in ASCII_TOKEN_RE.findall(text):
        terms.extend(_split_identifier(token))
    terms.extend(_cjk_bigrams(text))
    return terms


def build_sparse_terms(text: str) -> set[str]:
    return {
        term
        for term in tokenize_sparse_text(text)
        if term not in ENGLISH_QUERY_STOPWORDS
    }


@dataclass(frozen=True)
class SparseChunkRecord:
    chunk: Chunk
    path_tf: dict[str, int]
    symbol_tf: dict[str, int]
    content_tf: dict[str, int]
    path_len: int
    symbol_len: int
    content_len: int


def _counter_dict(text: str) -> tuple[dict[str, int], int]:
    tokens = tokenize_sparse_text(text)
    counts = Counter(tokens)
    return dict(counts), len(tokens)


def _build_record(chunk: Chunk) -> SparseChunkRecord:
    path_tf, path_len = _counter_dict(chunk.file_path)
    symbol_tf, symbol_len = _counter_dict(chunk.symbol_name or "")
    content_tf, content_len = _counter_dict(chunk.content)
    return SparseChunkRecord(
        chunk=chunk,
        path_tf=path_tf,
        symbol_tf=symbol_tf,
        content_tf=content_tf,
        path_len=path_len,
        symbol_len=symbol_len,
        content_len=content_len,
    )


def _idf(doc_freq: int, doc_count: int) -> float:
    return math.log(1.0 + ((doc_count - doc_freq + 0.5) / (doc_freq + 0.5)))


def _bm25_term_score(term_freq: int, *, doc_freq: int, doc_count: int, field_len: int, avg_field_len: float) -> float:
    if term_freq <= 0 or doc_freq <= 0 or doc_count <= 0:
        return 0.0
    normalized_avg = avg_field_len if avg_field_len > 0 else 1.0
    denominator = term_freq + BM25_K1 * (1.0 - BM25_B + BM25_B * (field_len / normalized_avg))
    return _idf(doc_freq, doc_count) * ((term_freq * (BM25_K1 + 1.0)) / denominator)


def _serialize_chunk(chunk: Chunk) -> dict[str, object]:
    return {
        "chunk_id": chunk.chunk_id,
        "file_path": chunk.file_path,
        "language": chunk.language,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "content": chunk.content,
        "chunk_type": chunk.chunk_type,
        "symbol_name": chunk.symbol_name,
        "source": chunk.source,
    }


def _deserialize_chunk(data: dict[str, object]) -> Chunk:
    return Chunk(
        chunk_id=str(data["chunk_id"]),
        file_path=str(data["file_path"]),
        language=str(data["language"]),
        start_line=int(data["start_line"]),
        end_line=int(data["end_line"]),
        content=str(data["content"]),
        chunk_type=str(data.get("chunk_type", "text_chunk")),
        symbol_name=data.get("symbol_name"),
        source=str(data.get("source", "codebase")),
    )


def _serialize_record(record: SparseChunkRecord) -> dict[str, object]:
    return {
        "chunk": _serialize_chunk(record.chunk),
        "path_tf": record.path_tf,
        "symbol_tf": record.symbol_tf,
        "content_tf": record.content_tf,
        "path_len": record.path_len,
        "symbol_len": record.symbol_len,
        "content_len": record.content_len,
    }


def _deserialize_record(data: dict[str, object]) -> SparseChunkRecord:
    return SparseChunkRecord(
        chunk=_deserialize_chunk(dict(data["chunk"])),
        path_tf={str(key): int(value) for key, value in dict(data["path_tf"]).items()},
        symbol_tf={str(key): int(value) for key, value in dict(data["symbol_tf"]).items()},
        content_tf={str(key): int(value) for key, value in dict(data["content_tf"]).items()},
        path_len=int(data["path_len"]),
        symbol_len=int(data["symbol_len"]),
        content_len=int(data["content_len"]),
    )


def _build_payload(repo_id: str, chunks: list[Chunk]) -> dict[str, object]:
    records = [_build_record(chunk) for chunk in chunks]
    doc_freqs = {"path": Counter(), "symbol": Counter(), "content": Counter()}
    total_lengths = {"path": 0, "symbol": 0, "content": 0}

    for record in records:
        total_lengths["path"] += record.path_len
        total_lengths["symbol"] += record.symbol_len
        total_lengths["content"] += record.content_len
        doc_freqs["path"].update(record.path_tf.keys())
        doc_freqs["symbol"].update(record.symbol_tf.keys())
        doc_freqs["content"].update(record.content_tf.keys())

    doc_count = len(records)
    avg_field_lengths = {
        field: (total_lengths[field] / doc_count if doc_count else 0.0)
        for field in total_lengths
    }
    return {
        "repo_id": repo_id,
        "doc_count": doc_count,
        "avg_field_lengths": avg_field_lengths,
        "doc_freqs": {
            field: dict(counter)
            for field, counter in doc_freqs.items()
        },
        "records": [_serialize_record(record) for record in records],
    }


def search_chunks(question: str, chunks: list[Chunk], top_k: int) -> list[SearchHit]:
    payload = _build_payload("adhoc", chunks)
    return _search_payload(question, payload, top_k)


def _search_payload(question: str, payload: dict[str, object], top_k: int) -> list[SearchHit]:
    query_terms = build_sparse_terms(question)
    if not query_terms:
        return []

    records = [_deserialize_record(record) for record in list(payload.get("records", []))]
    if not records:
        return []

    doc_count = int(payload.get("doc_count", 0))
    avg_field_lengths = dict(payload.get("avg_field_lengths", {}))
    raw_doc_freqs = dict(payload.get("doc_freqs", {}))
    doc_freqs = {
        field: {str(key): int(value) for key, value in dict(raw_doc_freqs.get(field, {})).items()}
        for field in FIELD_WEIGHTS
    }

    hits: list[SearchHit] = []
    for record in records:
        score = 0.0
        for term in query_terms:
            score += FIELD_WEIGHTS["path"] * _bm25_term_score(
                record.path_tf.get(term, 0),
                doc_freq=doc_freqs["path"].get(term, 0),
                doc_count=doc_count,
                field_len=record.path_len,
                avg_field_len=float(avg_field_lengths.get("path", 0.0)),
            )
            score += FIELD_WEIGHTS["symbol"] * _bm25_term_score(
                record.symbol_tf.get(term, 0),
                doc_freq=doc_freqs["symbol"].get(term, 0),
                doc_count=doc_count,
                field_len=record.symbol_len,
                avg_field_len=float(avg_field_lengths.get("symbol", 0.0)),
            )
            score += FIELD_WEIGHTS["content"] * _bm25_term_score(
                record.content_tf.get(term, 0),
                doc_freq=doc_freqs["content"].get(term, 0),
                doc_count=doc_count,
                field_len=record.content_len,
                avg_field_len=float(avg_field_lengths.get("content", 0.0)),
            )
        if score <= 0:
            continue
        hits.append(SearchHit(chunk=record.chunk, score=score))

    if not hits:
        return []

    hits.sort(key=lambda hit: hit.score, reverse=True)
    max_score = hits[0].score
    return [
        SearchHit(chunk=hit.chunk, score=hit.score / max_score)
        for hit in hits[:top_k]
    ]


class SparseIndexStore:
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)

    def index_path(self, repo_id: str) -> Path:
        safe_repo_id = repo_id.replace("/", "_").replace(" ", "_")
        return self.base_path / f"{safe_repo_id}__sparse.json"

    def reset(self, repo_id: str) -> None:
        path = self.index_path(repo_id)
        if path.exists():
            path.unlink()

    def upsert(self, repo_id: str, chunks: list[Chunk]) -> None:
        payload = _build_payload(repo_id, chunks)
        self.index_path(repo_id).write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )

    def search(self, repo_id: str, question: str, top_k: int) -> list[SearchHit]:
        path = self.index_path(repo_id)
        if not path.exists():
            return []
        payload = json.loads(path.read_text(encoding="utf-8"))
        return _search_payload(question, payload, top_k)

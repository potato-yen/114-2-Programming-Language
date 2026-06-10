from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from codebase_rag.config import Settings
from codebase_rag.indexer import build_index
from codebase_rag.llm import OpenRouterClient
from codebase_rag.manifest import IndexManifestStore
from codebase_rag.prompt import build_answer_prompt
from codebase_rag.repository import build_repo_tree, make_repo_id
from codebase_rag.retriever import (
    build_abstain_message,
    format_retrieval_only_answer,
    render_retrieved_context,
    search_codebase,
    should_abstain,
)

load_dotenv()

app = FastAPI(title="Codebase Evidence Explorer API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RepoPathRequest(BaseModel):
    repo_path: str


class QueryRequest(RepoPathRequest):
    question: str
    use_llm: bool = True


class FileRequest(RepoPathRequest):
    file_path: str


def _repo_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser().resolve()
    if not path.is_dir():
        raise HTTPException(status_code=400, detail=f"Repository path is not a directory: {path}")
    return path


def _settings() -> Settings:
    return Settings.from_env()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/repo/status")
def repo_status(request: RepoPathRequest) -> dict[str, object]:
    repo_path = _repo_path(request.repo_path)
    settings = _settings()
    manifest = IndexManifestStore(settings.vector_db_path).read(repo_path)
    return {
        "repo_id": make_repo_id(repo_path),
        "repo_path": str(repo_path),
        "tree": build_repo_tree(repo_path, settings),
        "index_available": manifest is not None,
        "manifest": manifest,
    }


@app.post("/api/repo/index")
def repo_index(request: RepoPathRequest) -> dict[str, object]:
    repo_path = _repo_path(request.repo_path)
    try:
        stats = build_index(repo_path=repo_path, settings=_settings(), reset=True)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "repo_id": stats.repo_id,
        "file_count": stats.file_count,
        "chunk_count": stats.chunk_count,
    }


@app.post("/api/query")
def query(request: QueryRequest) -> dict[str, object]:
    repo_path = _repo_path(request.repo_path)
    settings = _settings()
    try:
        hits = search_codebase(repo_path, request.question, settings)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    abstained = should_abstain(request.question, hits)
    answer_notice = None
    if abstained:
        answer = build_abstain_message(request.question)
        answer_mode = "abstained"
    elif request.use_llm and settings.openrouter_api_key:
        try:
            answer = OpenRouterClient(
                api_key=settings.openrouter_api_key,
                model=settings.openrouter_model,
            ).answer(build_answer_prompt(request.question, render_retrieved_context(hits)))
            answer_mode = "llm"
        except Exception:
            answer = format_retrieval_only_answer(request.question, hits)
            answer_mode = "retrieval"
            answer_notice = "LLM request failed, so this answer uses retrieval results only."
    else:
        answer = format_retrieval_only_answer(request.question, hits)
        answer_mode = "retrieval"
        if request.use_llm:
            answer_notice = "LLM answer requested, but OPENROUTER_API_KEY is not configured."

    return {
        "answer": answer,
        "abstained": abstained,
        "answer_mode": answer_mode,
        "answer_notice": answer_notice,
        "sources": [
            {
                "file_path": hit.chunk.file_path,
                "language": hit.chunk.language,
                "start_line": hit.chunk.start_line,
                "end_line": hit.chunk.end_line,
                "symbol_name": hit.chunk.symbol_name,
                "score": hit.score,
                "content": hit.chunk.content,
            }
            for hit in hits
        ],
    }


@app.post("/api/file")
def read_file(request: FileRequest) -> dict[str, object]:
    repo_path = _repo_path(request.repo_path)
    target = (repo_path / request.file_path).resolve()
    if target != repo_path and repo_path not in target.parents:
        raise HTTPException(status_code=400, detail="File path escapes repository root")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    try:
        content = target.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="File is not UTF-8 text") from exc
    return {
        "file_path": target.relative_to(repo_path).as_posix(),
        "content": content,
    }

# HW4 RAG

Codebase RAG for cross-language code indexing and grounded code-navigation questions, with both a CLI and a local evidence-explorer frontend.

This project indexes a local repository, retrieves grounded code chunks, and answers questions such as:

- `Where is the login feature?`
- `Where is markdown rendering implemented?`
- `Where is LaTeX compilation handled?`

It is designed as a course assignment MVP, but has been extended with real-repo testing and hybrid retrieval.

## What It Does

- scans a target repository and filters common noise directories/files
- chunks code with symbol-aware splitting plus line-based fallback
- builds a dense embedding index with Chroma
- builds a sparse BM25-like sidecar index for lexical retrieval
- fuses dense and sparse results for hybrid retrieval
- returns grounded file paths, symbols, and line ranges
- abstains when evidence is too weak or the question is unrelated
- optionally uses OpenRouter to turn retrieved context into a natural-language answer
- provides a local web workbench for repository trees, reusable indexes, ranked evidence, and read-only source inspection

## Retrieval Design

The current pipeline is:

```text
repo path
-> scan files
-> chunk with metadata
-> dense embeddings + Chroma
-> sparse sidecar index
-> user query
-> dense retrieval + sparse retrieval
-> reciprocal-rank-style fusion
-> light rerank
-> abstain or answer
```

Key design choices:

- `intfloat/multilingual-e5-small` for multilingual query-to-code embedding
- symbol-aware chunking to avoid mixing unrelated functions in one chunk
- hybrid retrieval to combine semantic and exact lexical evidence
- deterministic abstention to avoid answering unrelated questions

## Installation

```bash
uv venv
source .venv/bin/activate
uv sync
cd frontend
npm install
cd ..
```

The first index build downloads the embedding model into the Hugging Face user cache. Later runs reuse the cached model.

## Environment

Create a `.env` file:

```env
OPENROUTER_API_KEY=...
OPENROUTER_MODEL=openai/gpt-oss-120b:free
EMBEDDING_MODEL=intfloat/multilingual-e5-small
VECTOR_DB_PATH=./vector_db
TOP_K=6
```

`OPENROUTER_API_KEY` is optional. If it is missing, the project still works in retrieval-only mode.

## Usage

The Web workbench and CLI use the same persistent indexes under `VECTOR_DB_PATH`. An index built by either interface can be reused by the other.

### Option A: Web Workbench

Start the API from the repository root:

```bash
uv run python web_api.py
```

In another terminal, start the frontend:

```bash
cd frontend
npm run dev
```

Open `http://127.0.0.1:5173`, then:

1. Enter the absolute path of the repository to inspect.
2. Click the arrow button to scan its filtered project tree and check index availability.
3. Select `Use existing` to reuse a previous index, or `Rebuild index` after source changes.
4. Ask a code-navigation question.
5. Optionally enable `LLM answer` to format the retrieved evidence with OpenRouter.
6. Click a ranked source to inspect its file and highlighted line range.

`Rebuild index` rescans files and regenerates embeddings. `Use existing` avoids that work and starts querying the stored index immediately.

### Option B: CLI

Build or rebuild an index:

```bash
uv run python index_repo.py /path/to/your/project --reset
```

Example:

```bash
uv run python index_repo.py ./sample_project --reset
```

Ask with an OpenRouter-formatted answer:

```bash
uv run python ask.py "Where is the login feature?" --repo-path /path/to/your/project
```

Ask in retrieval-only mode:

```bash
uv run python ask.py "Where is the login feature?" --repo-path /path/to/your/project --no-llm
```

Example:

```bash
uv run python ask.py "登入功能在哪裡？" --repo-path ./sample_project --no-llm
```

The CLI loads the local embedding model for each command, so repeated questions are slower than using the long-running Web API. If the system cannot find enough evidence, it returns an abstention message instead of forcing an answer.

## Output Style

Retrieval-only mode returns grounded results like:

```text
可能相關位置：

1. frontend/src/pages/LoginPage.tsx, LoginPage(), lines 1-112
2. frontend/src/markdownRenderer.ts, lines 1-113
```

LLM mode uses the same retrieved evidence, but formats it into a more natural answer.

Ranked source scores are relative retrieval-order signals produced by hybrid fusion. They are not probabilities or guarantees that a source answers the question.

## Project Structure

- `index_repo.py`: build or rebuild an index for one repo
- `ask.py`: ask questions against an indexed repo
- `codebase_rag/scanner.py`: repo scanning and filtering
- `codebase_rag/chunker.py`: chunking and chunk rendering
- `codebase_rag/symbols.py`: regex-based symbol extraction
- `codebase_rag/vectorstore.py`: Chroma dense index
- `codebase_rag/sparse_index.py`: sparse BM25-like sidecar index
- `codebase_rag/retriever.py`: hybrid retrieval, rerank, abstention
- `codebase_rag/llm.py`: OpenRouter client
- `codebase_rag/api.py`: local FastAPI endpoints for indexing, querying, and file inspection
- `frontend/`: React evidence-explorer workbench

## Verification

Run the test suite:

```bash
uv run pytest -v
```

## Current Scope

This is still a compact assignment project, not a production code search system.

Current strengths:

- works on multiple real repositories
- handles both retrieval-only and LLM-assisted flows
- better lexical retrieval than the original dense-only MVP
- reuses indexes by stable absolute-path-based repository IDs

Current limitations:

- Chinese feature names to English backend identifiers are still imperfect
- retrieval quality depends on repo structure and naming quality
- file-level reranking can over-prioritize files containing many weak lexical matches
- cross-module behavior, indirect dependencies, and feature-absence questions remain difficult
- ranked retrieval scores are relative ordering signals, not evidence confidence
- the frontend is a local development tool and does not provide authentication or remote deployment hardening

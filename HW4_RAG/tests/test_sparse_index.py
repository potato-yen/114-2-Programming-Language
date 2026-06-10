from pathlib import Path

from codebase_rag.sparse_index import SparseIndexStore
from codebase_rag.types import Chunk


def test_sparse_index_prefers_distinct_query_concept_coverage(tmp_path):
    store = SparseIndexStore(tmp_path)
    repo_id = "Lazy-Git"
    chunks = [
        Chunk(
            chunk_id="src/git_adapter.py:29-45",
            file_path="src/git_adapter.py",
            language="python",
            start_line=29,
            end_line=45,
            content="def list_local_branches(): return []",
            chunk_type="function",
            symbol_name="list_local_branches",
        ),
        Chunk(
            chunk_id="src/git_adapter.py:72-79",
            file_path="src/git_adapter.py",
            language="python",
            start_line=72,
            end_line=79,
            content="def list_branches(): return list_local_branches() + list_remote_branches()",
            chunk_type="function",
            symbol_name="list_branches",
        ),
        Chunk(
            chunk_id="src/lazygit.py:58-135",
            file_path="src/lazygit.py",
            language="python",
            start_line=58,
            end_line=135,
            content="class Cache: def get_cached_local_branches(self): return self.local_branches_cache",
            chunk_type="class",
            symbol_name="Cache",
        ),
    ]

    store.reset(repo_id)
    store.upsert(repo_id, chunks)

    hits = store.search(repo_id, "Where is the branch cache?", top_k=3)

    assert hits[0].chunk.file_path == "src/lazygit.py"


def test_sparse_index_persists_sidecar_per_repo(tmp_path):
    store = SparseIndexStore(tmp_path)
    repo_id = "repo"
    chunks = [
        Chunk(
            chunk_id="src/auth.py:1-10",
            file_path="src/auth.py",
            language="python",
            start_line=1,
            end_line=10,
            content="def login(): return True",
            chunk_type="function",
            symbol_name="login",
        )
    ]

    store.reset(repo_id)
    store.upsert(repo_id, chunks)

    assert store.index_path(repo_id).exists()
    loaded_hits = store.search(repo_id, "login", top_k=1)

    assert loaded_hits[0].chunk.file_path == "src/auth.py"

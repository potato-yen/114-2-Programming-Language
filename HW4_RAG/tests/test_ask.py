from pathlib import Path

from codebase_rag.config import Settings
from codebase_rag.retriever import (
    build_sparse_terms,
    format_retrieval_only_answer,
    group_hits_by_file,
    render_retrieved_context,
    search_codebase,
    rerank_hits,
    sparse_search,
    should_abstain,
)
from codebase_rag.repository import make_repo_id
from codebase_rag.sparse_index import SparseIndexStore
from codebase_rag.types import Chunk, SearchHit


def build_hit(file_path: str, symbol_name: str | None, score: float) -> SearchHit:
    return SearchHit(
        chunk=Chunk(
            chunk_id=f"{file_path}:1-10",
            file_path=file_path,
            language="python",
            start_line=1,
            end_line=10,
            content="def login():\n    return True",
            symbol_name=symbol_name,
        ),
        score=score,
    )


def test_retrieval_only_answer_includes_grounded_metadata():
    hits = [
        build_hit("src/auth.py", "login", 0.98),
        build_hit("src/db.py", None, 0.71),
    ]

    answer = format_retrieval_only_answer("登入功能在哪裡？", hits)

    assert "可能相關位置" in answer
    assert "src/auth.py" in answer
    assert "lines 1-10" in answer
    assert "login" in answer


def test_group_hits_by_file_merges_duplicate_paths():
    hits = [
        SearchHit(
            chunk=Chunk(
                chunk_id="src/auth.py:1-10",
                file_path="src/auth.py",
                language="python",
                start_line=1,
                end_line=10,
                content="def login():\n    return True",
                symbol_name="login",
            ),
            score=0.98,
        ),
        SearchHit(
            chunk=Chunk(
                chunk_id="src/auth.py:21-30",
                file_path="src/auth.py",
                language="python",
                start_line=21,
                end_line=30,
                content="def logout():\n    return None",
                symbol_name="logout",
            ),
            score=0.77,
        ),
        SearchHit(
            chunk=Chunk(
                chunk_id="src/db.py:1-8",
                file_path="src/db.py",
                language="python",
                start_line=1,
                end_line=8,
                content="def connect_db():\n    return 'db'",
                symbol_name="connect_db",
            ),
            score=0.71,
        ),
    ]

    grouped = group_hits_by_file(hits)

    assert len(grouped) == 2
    assert grouped[0].file_path == "src/auth.py"
    assert grouped[0].line_ranges == [(1, 10), (21, 30)]
    assert grouped[0].symbol_names == ["login", "logout"]


def test_retrieval_only_answer_lists_each_file_once():
    hits = [
        SearchHit(
            chunk=Chunk(
                chunk_id="src/auth.py:1-10",
                file_path="src/auth.py",
                language="python",
                start_line=1,
                end_line=10,
                content="def login():\n    return True",
                symbol_name="login",
            ),
            score=0.98,
        ),
        SearchHit(
            chunk=Chunk(
                chunk_id="src/auth.py:21-30",
                file_path="src/auth.py",
                language="python",
                start_line=21,
                end_line=30,
                content="def logout():\n    return None",
                symbol_name="logout",
            ),
            score=0.77,
        ),
    ]

    answer = format_retrieval_only_answer("登入功能在哪裡？", hits)

    assert answer.count("src/auth.py") == 1
    assert "lines 1-10, 21-30" in answer


def test_render_retrieved_context_groups_chunks_by_file():
    hits = [
        SearchHit(
            chunk=Chunk(
                chunk_id="src/auth.py:1-10",
                file_path="src/auth.py",
                language="python",
                start_line=1,
                end_line=10,
                content="def login():\n    return True",
                symbol_name="login",
            ),
            score=0.98,
        ),
        SearchHit(
            chunk=Chunk(
                chunk_id="src/auth.py:21-30",
                file_path="src/auth.py",
                language="python",
                start_line=21,
                end_line=30,
                content="def logout():\n    return None",
                symbol_name="logout",
            ),
            score=0.77,
        ),
    ]

    context = render_retrieved_context(hits)

    assert context.count("File: src/auth.py") == 1
    assert "Lines: 1-10, 21-30" in context
    assert "def login()" in context
    assert "def logout()" in context


def test_settings_without_api_key_can_still_answer(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    settings = Settings.from_env()

    assert settings.openrouter_api_key is None


def test_rerank_hits_prefers_auth_code_over_docs_noise():
    hits = [
        SearchHit(
            chunk=Chunk(
                chunk_id="docs/spec.md:1-10",
                file_path="docs/spec.md",
                language="markdown",
                start_line=1,
                end_line=10,
                content="登入流程規劃與 UI 文案。",
            ),
            score=0.92,
        ),
        SearchHit(
            chunk=Chunk(
                chunk_id="lib/errors/messages.ts:1-10",
                file_path="lib/errors/messages.ts",
                language="typescript",
                start_line=1,
                end_line=10,
                content="INVALID_CREDENTIALS: 帳號或密碼錯誤",
                symbol_name="getErrorMessage",
            ),
            score=0.9,
        ),
        SearchHit(
            chunk=Chunk(
                chunk_id="lib/auth/service.ts:50-60",
                file_path="lib/auth/service.ts",
                language="typescript",
                start_line=50,
                end_line=60,
                content="export async function signIn(sb, input) { return sb.auth.signInWithPassword(input); }",
                chunk_type="function",
                symbol_name="signIn",
            ),
            score=0.78,
        ),
        SearchHit(
            chunk=Chunk(
                chunk_id="app/(auth)/sign-in.tsx:1-20",
                file_path="app/(auth)/sign-in.tsx",
                language="typescriptreact",
                start_line=1,
                end_line=20,
                content="Button title='登入' onPress={onSubmit}; await signIn(getSupabaseClient(), { email, password: pwd });",
                chunk_type="function",
                symbol_name="SignIn",
            ),
            score=0.76,
        ),
    ]

    reranked = rerank_hits("登入功能在哪裡？", hits)

    assert reranked[0].chunk.file_path == "app/(auth)/sign-in.tsx"
    assert reranked[1].chunk.file_path == "lib/auth/service.ts"


def test_rerank_hits_prefers_file_with_multiple_supporting_chunks():
    hits = [
        SearchHit(
            chunk=Chunk(
                chunk_id="app/(app)/territory/map.tsx:320-360",
                file_path="app/(app)/territory/map.tsx",
                language="typescriptreact",
                start_line=320,
                end_line=360,
                content="畫面顯示國庫、財政資訊與領地排行榜。",
                chunk_type="function",
                symbol_name="TerritoryMap",
            ),
            score=0.84,
        ),
        SearchHit(
            chunk=Chunk(
                chunk_id="lib/territory/tax.ts:1-20",
                file_path="lib/territory/tax.ts",
                language="typescript",
                start_line=1,
                end_line=20,
                content="國庫稅收結算與財政分配規則。",
                chunk_type="function",
                symbol_name="runTreasuryTax",
            ),
            score=0.65,
        ),
        SearchHit(
            chunk=Chunk(
                chunk_id="lib/territory/tax.ts:21-40",
                file_path="lib/territory/tax.ts",
                language="typescript",
                start_line=21,
                end_line=40,
                content="更新國庫 ledger 與財政 settlement 統計。",
                chunk_type="function",
                symbol_name="applyTreasurySettlement",
            ),
            score=0.62,
        ),
    ]

    reranked = rerank_hits("國庫財政機制在哪裡？", hits)

    assert reranked[0].chunk.file_path == "lib/territory/tax.ts"
    assert reranked[1].chunk.file_path == "lib/territory/tax.ts"


def test_build_sparse_terms_supports_cjk_bigrams_and_identifiers():
    terms = build_sparse_terms("國庫財政機制 treasuryTax signInWithPassword")

    assert "國庫" in terms
    assert "財政" in terms
    assert "treasury" in terms
    assert "tax" in terms
    assert "signin" in terms
    assert "password" in terms


def test_build_sparse_terms_excludes_english_query_stopwords():
    terms = build_sparse_terms("How is the branch cache implemented?")

    assert "branch" in terms
    assert "cache" in terms
    assert "how" not in terms
    assert "is" not in terms
    assert "the" not in terms


def test_build_sparse_terms_normalizes_plural_identifier_parts():
    terms = build_sparse_terms("local_branches_cache")

    assert "branch" in terms
    assert "cache" in terms


def test_rerank_hits_prefers_concept_coverage_over_many_stopword_matches():
    hits = [
        SearchHit(
            chunk=Chunk(
                chunk_id="src/git_adapter.py:1-10",
                file_path="src/git_adapter.py",
                language="python",
                start_line=1,
                end_line=10,
                content="def is_git_repo(): return True",
                symbol_name="is_git_repo",
            ),
            score=0.95,
        ),
        SearchHit(
            chunk=Chunk(
                chunk_id="src/git_adapter.py:11-20",
                file_path="src/git_adapter.py",
                language="python",
                start_line=11,
                end_line=20,
                content="def new_branch(name): return name",
                symbol_name="new_branch",
            ),
            score=0.94,
        ),
        SearchHit(
            chunk=Chunk(
                chunk_id="src/git_adapter.py:21-30",
                file_path="src/git_adapter.py",
                language="python",
                start_line=21,
                end_line=30,
                content="def switch(branch): return branch",
                symbol_name="switch",
            ),
            score=0.93,
        ),
        SearchHit(
            chunk=Chunk(
                chunk_id="src/lazygit.py:58-135",
                file_path="src/lazygit.py",
                language="python",
                start_line=58,
                end_line=135,
                content="class Cache: local_branches_cache = []",
                symbol_name="Cache",
                chunk_type="class",
            ),
            score=0.8,
        ),
    ]

    reranked = rerank_hits("How is the branch cache implemented?", hits)

    assert reranked[0].chunk.file_path == "src/lazygit.py"


def test_sparse_search_prefers_exact_keyword_matches():
    chunks = [
        Chunk(
            chunk_id="app/map.tsx:1-20",
            file_path="app/map.tsx",
            language="typescriptreact",
            start_line=1,
            end_line=20,
            content="畫面顯示國庫資訊與排行榜。",
        ),
        Chunk(
            chunk_id="lib/territory/tax.ts:1-20",
            file_path="lib/territory/tax.ts",
            language="typescript",
            start_line=1,
            end_line=20,
            content="apply treasury tax settlement for territory groups",
            symbol_name="runTreasuryTax",
            chunk_type="function",
        ),
    ]

    hits = sparse_search("國庫財政機制在哪裡？", chunks, top_k=5)

    assert hits[0].chunk.file_path == "app/map.tsx"


def test_search_codebase_uses_wide_dense_candidate_pool(monkeypatch):
    class FakeEmbedder:
        def embed_query(self, text: str) -> list[float]:
            return [1.0, 0.0, 0.0]

    class FakeVectorStore:
        def __init__(self, base_path):
            self.base_path = base_path

        def query(self, repo_id: str, query_embedding: list[float], top_k: int) -> list[SearchHit]:
            _ = repo_id
            _ = query_embedding
            common_hits = [
                SearchHit(
                    chunk=Chunk(
                        chunk_id="app/(auth)/sign-in.tsx:1-10",
                        file_path="app/(auth)/sign-in.tsx",
                        language="typescriptreact",
                        start_line=1,
                        end_line=10,
                        content="await signIn(getSupabaseClient(), { email, password: pwd })",
                        chunk_type="function",
                        symbol_name="onSubmit",
                    ),
                    score=0.78,
                )
            ]
            if top_k < 100:
                return common_hits
            return common_hits + [
                SearchHit(
                    chunk=Chunk(
                        chunk_id="lib/auth/service.ts:1-80",
                        file_path="lib/auth/service.ts",
                        language="typescript",
                        start_line=1,
                        end_line=80,
                        content="export async function signIn(sb, input) { return sb.auth.signInWithPassword(input); }",
                        chunk_type="function",
                        symbol_name="signIn",
                    ),
                    score=0.72,
                )
            ]

        def list_chunks(self, repo_id: str) -> list[Chunk]:
            _ = repo_id
            return [
                Chunk(
                    chunk_id="app/(auth)/sign-in.tsx:1-10",
                    file_path="app/(auth)/sign-in.tsx",
                    language="typescriptreact",
                    start_line=1,
                    end_line=10,
                    content="await signIn(getSupabaseClient(), { email, password: pwd })",
                    chunk_type="function",
                    symbol_name="onSubmit",
                ),
            ]

    monkeypatch.setattr("codebase_rag.retriever.ChromaVectorStore", FakeVectorStore)

    settings = Settings.from_env()
    hits = search_codebase(
        repo_path=Path("/tmp/Linquest"),
        question="登入功能在哪裡？",
        settings=settings,
        embedder=FakeEmbedder(),
        vector_db_path=Path("/tmp/vector_db"),
    )

    top_two = {hit.chunk.file_path for hit in hits[:2]}

    assert "app/(auth)/sign-in.tsx" in top_two
    assert "lib/auth/service.ts" in top_two


def test_search_codebase_uses_sparse_sidecar_for_hybrid_ranking(tmp_path, monkeypatch):
    class FakeEmbedder:
        def embed_query(self, text: str) -> list[float]:
            return [1.0, 0.0, 0.0]

    class FakeVectorStore:
        def __init__(self, base_path):
            self.base_path = base_path

        def query(self, repo_id: str, query_embedding: list[float], top_k: int) -> list[SearchHit]:
            _ = repo_id
            _ = query_embedding
            _ = top_k
            return [
                SearchHit(
                    chunk=Chunk(
                        chunk_id="src/git_adapter.py:29-45",
                        file_path="src/git_adapter.py",
                        language="python",
                        start_line=29,
                        end_line=45,
                        content="def list_local_branches(): return []",
                        chunk_type="function",
                        symbol_name="list_local_branches",
                    ),
                    score=0.88,
                ),
                SearchHit(
                    chunk=Chunk(
                        chunk_id="src/git_adapter.py:72-79",
                        file_path="src/git_adapter.py",
                        language="python",
                        start_line=72,
                        end_line=79,
                        content="def list_branches(): return list_local_branches() + list_remote_branches()",
                        chunk_type="function",
                        symbol_name="list_branches",
                    ),
                    score=0.84,
                ),
            ]

        def list_chunks(self, repo_id: str) -> list[Chunk]:
            _ = repo_id
            return []

    monkeypatch.setattr("codebase_rag.retriever.ChromaVectorStore", FakeVectorStore)

    repo_path = tmp_path / "Lazy-Git"
    repo_path.mkdir()
    vector_db_path = tmp_path / "vector_db"
    sparse_index = SparseIndexStore(vector_db_path)
    sparse_index.upsert(
        make_repo_id(repo_path),
        [
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
                chunk_id="src/lazygit.py:58-135",
                file_path="src/lazygit.py",
                language="python",
                start_line=58,
                end_line=135,
                content="class Cache: def get_cached_local_branches(self): return self.local_branches_cache",
                chunk_type="class",
                symbol_name="Cache",
            ),
        ],
    )

    settings = Settings.from_env()
    hits = search_codebase(
        repo_path=repo_path,
        question="Where is the branch cache?",
        settings=settings,
        embedder=FakeEmbedder(),
        vector_db_path=vector_db_path,
    )

    assert hits[0].chunk.file_path == "src/lazygit.py"


def test_should_abstain_for_unrelated_question_without_overlap():
    hits = [
        SearchHit(
            chunk=Chunk(
                chunk_id="app/roadmap.tsx:1-20",
                file_path="app/roadmap.tsx",
                language="typescriptreact",
                start_line=1,
                end_line=20,
                content="render roadmap stage progress and milestone cards",
            ),
            score=0.68,
        ),
        SearchHit(
            chunk=Chunk(
                chunk_id="lib/battle.ts:1-20",
                file_path="lib/battle.ts",
                language="typescript",
                start_line=1,
                end_line=20,
                content="submit battle answer and resolve battle state",
            ),
            score=0.66,
        ),
    ]

    assert should_abstain("今天天氣如何？", hits) is True

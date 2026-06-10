from codebase_rag.vectorstore import ChromaVectorStore
from tests.helpers import make_chunk


def test_vectorstore_round_trip(tmp_path):
    store = ChromaVectorStore(base_path=tmp_path / "vector_db")
    chunks = [
        make_chunk("src/auth.py", 1, 10, "def login():\n    return True"),
        make_chunk("src/db.py", 1, 8, "def connect_db():\n    return 'db'"),
    ]
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    store.reset("sample-project")
    store.upsert("sample-project", chunks, embeddings)
    hits = store.query("sample-project", [1.0, 0.0, 0.0], top_k=2)

    assert hits[0].chunk.file_path == "src/auth.py"
    assert hits[0].chunk.start_line == 1
    assert len(hits) == 2

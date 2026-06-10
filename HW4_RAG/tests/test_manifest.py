from codebase_rag.manifest import IndexManifestStore


def test_manifest_store_round_trip(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    store = IndexManifestStore(tmp_path / "vector_db")

    store.write(
        repo_path=repo,
        repo_id="repo-123",
        file_count=3,
        chunk_count=7,
        embedding_model="test-model",
    )

    manifest = store.read(repo)

    assert manifest is not None
    assert manifest.repo_id == "repo-123"
    assert manifest.file_count == 3
    assert manifest.chunk_count == 7
    assert manifest.embedding_model == "test-model"

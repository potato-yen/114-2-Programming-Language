from codebase_rag.config import Settings
from codebase_rag.repository import build_repo_tree, make_repo_id


def test_make_repo_id_distinguishes_same_named_directories(tmp_path):
    first = tmp_path / "one" / "project"
    second = tmp_path / "two" / "project"
    first.mkdir(parents=True)
    second.mkdir(parents=True)

    assert make_repo_id(first) != make_repo_id(second)
    assert make_repo_id(first).startswith("project-")


def test_build_repo_tree_uses_scanner_exclusions(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "node_modules").mkdir()
    (repo / "src" / "app.py").write_text("print('hello')", encoding="utf-8")
    (repo / "node_modules" / "noise.js").write_text("noise", encoding="utf-8")
    (repo / "README.md").write_text("# Repo", encoding="utf-8")

    tree = build_repo_tree(repo, Settings.from_env())

    assert tree["name"] == "repo"
    assert tree["type"] == "directory"
    rendered = str(tree)
    assert "src" in rendered
    assert "app.py" in rendered
    assert "README.md" in rendered
    assert "node_modules" not in rendered

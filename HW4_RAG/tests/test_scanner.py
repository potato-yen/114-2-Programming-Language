from pathlib import Path

from codebase_rag.config import Settings
from codebase_rag.language import detect_language
from codebase_rag.scanner import scan_repository


def test_detect_language_maps_known_extensions():
    assert detect_language(Path("app.py")) == "python"
    assert detect_language(Path("auth.ts")) == "typescript"
    assert detect_language(Path("README.md")) == "markdown"
    assert detect_language(Path("archive.zip")) is None


def test_scan_repository_skips_excluded_dirs_and_binary_files(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "node_modules").mkdir()
    (repo / "src" / "app.py").write_text("print('hello')\n", encoding="utf-8")
    (repo / "src" / "view.ts").write_text("export const view = 1;\n", encoding="utf-8")
    (repo / "node_modules" / "ignored.js").write_text(
        "console.log('skip');\n",
        encoding="utf-8",
    )
    (repo / "src" / "image.png").write_bytes(b"\x89PNG\r\n")

    settings = Settings.from_env()
    files = scan_repository(repo, settings)

    assert [item.rel_path for item in files] == ["src/app.py", "src/view.ts"]
    assert files[0].language == "python"
    assert files[1].language == "typescript"


def test_scan_repository_skips_lockfiles_even_when_json(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "src" / "app.ts").write_text("export const app = true;\n", encoding="utf-8")
    (repo / "package-lock.json").write_text('{"name": "demo"}\n', encoding="utf-8")

    files = scan_repository(repo, Settings.from_env())

    assert [item.rel_path for item in files] == ["src/app.ts"]

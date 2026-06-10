from pathlib import Path

from codebase_rag.chunker import chunk_source_file, render_chunk_document
from codebase_rag.config import Settings
from codebase_rag.types import SourceFile


def build_source_file(line_count: int) -> SourceFile:
    content = "\n".join(f"line {index}" for index in range(1, line_count + 1))
    return SourceFile(
        abs_path=Path("/tmp/example.py"),
        rel_path="src/example.py",
        language="python",
        content=content,
    )


def test_chunk_source_file_uses_overlap():
    settings = Settings.from_env()
    source_file = build_source_file(150)

    chunks = chunk_source_file(source_file, settings)

    assert len(chunks) == 2
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 80
    assert chunks[1].start_line == 71
    assert chunks[1].end_line == 150


def test_render_chunk_document_includes_metadata_header():
    settings = Settings.from_env()
    source_file = build_source_file(5)

    chunk = chunk_source_file(source_file, settings)[0]
    document = render_chunk_document(chunk)

    assert "File: src/example.py" in document
    assert "Language: python" in document
    assert "Lines: 1-5" in document
    assert "Code:" in document


def test_chunk_source_file_splits_top_level_symbols_into_separate_chunks():
    source_file = SourceFile(
        abs_path=Path("/tmp/service.ts"),
        rel_path="lib/auth/service.ts",
        language="typescript",
        content=(
            "export interface SignInInput {\n"
            "  email: string;\n"
            "  password: string;\n"
            "}\n\n"
            "export async function signUp(sb, input) {\n"
            "  return sb.auth.signUp(input);\n"
            "}\n\n"
            "export async function signIn(sb, input) {\n"
            "  return sb.auth.signInWithPassword(input);\n"
            "}\n"
        ),
    )

    chunks = chunk_source_file(source_file, Settings.from_env())

    assert [chunk.symbol_name for chunk in chunks] == [None, "signUp", "signIn"]
    assert chunks[1].start_line == 6
    assert chunks[1].end_line == 9
    assert chunks[2].start_line == 10
    assert chunks[2].end_line == 12

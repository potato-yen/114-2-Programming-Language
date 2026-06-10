from pathlib import Path

from codebase_rag.chunker import chunk_source_file
from codebase_rag.config import Settings
from codebase_rag.types import SourceFile


def test_python_chunk_gets_symbol_name():
    source_file = SourceFile(
        abs_path=Path("/tmp/auth.py"),
        rel_path="src/auth.py",
        language="python",
        content="def login(username, password):\n    return True\n",
    )

    chunk = chunk_source_file(source_file, Settings.from_env())[0]

    assert chunk.symbol_name == "login"
    assert chunk.chunk_type == "function"


def test_typescript_chunk_gets_symbol_name():
    source_file = SourceFile(
        abs_path=Path("/tmp/auth.ts"),
        rel_path="src/auth.ts",
        language="typescript",
        content="export function postLogin(username, password) {\n  return true;\n}\n",
    )

    chunk = chunk_source_file(source_file, Settings.from_env())[0]

    assert chunk.symbol_name == "postLogin"
    assert chunk.chunk_type == "function"


def test_typescript_react_prefers_export_default_component_name():
    source_file = SourceFile(
        abs_path=Path("/tmp/sign-in.tsx"),
        rel_path="app/(auth)/sign-in.tsx",
        language="typescriptreact",
        content=(
            "export default function SignIn() {\n"
            "  function onSubmit() {\n"
            "    return true;\n"
            "  }\n"
            "  return onSubmit();\n"
            "}\n"
        ),
    )

    chunk = chunk_source_file(source_file, Settings.from_env())[0]

    assert chunk.symbol_name == "SignIn"
    assert chunk.chunk_type == "function"

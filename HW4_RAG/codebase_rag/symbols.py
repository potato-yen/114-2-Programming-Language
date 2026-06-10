from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SymbolSpan:
    chunk_type: str
    symbol_name: str
    start_line: int
    end_line: int


PATTERNS: dict[str, list[tuple[str, re.Pattern[str]]]] = {
    "python": [
        ("function", re.compile(r"^def\s+(\w+)\s*\(", re.MULTILINE)),
        ("class", re.compile(r"^class\s+(\w+)", re.MULTILINE)),
    ],
    "javascript": [
        ("function", re.compile(r"^export\s+default\s+function\s+(\w+)\s*\(", re.MULTILINE)),
        ("function", re.compile(r"^export\s+function\s+(\w+)\s*\(", re.MULTILINE)),
        ("function", re.compile(r"^function\s+(\w+)\s*\(", re.MULTILINE)),
        ("function", re.compile(r"^export\s+const\s+(\w+)\s*=\s*(?:async\s*)?\(", re.MULTILINE)),
        ("function", re.compile(r"^const\s+(\w+)\s*=\s*(?:async\s*)?\(", re.MULTILINE)),
        ("class", re.compile(r"^export\s+class\s+(\w+)", re.MULTILINE)),
        ("class", re.compile(r"^class\s+(\w+)", re.MULTILINE)),
    ],
    "javascriptreact": [
        ("function", re.compile(r"^export\s+default\s+function\s+(\w+)\s*\(", re.MULTILINE)),
        ("function", re.compile(r"^export\s+function\s+(\w+)\s*\(", re.MULTILINE)),
        ("function", re.compile(r"^function\s+(\w+)\s*\(", re.MULTILINE)),
        ("function", re.compile(r"^export\s+const\s+(\w+)\s*=\s*(?:async\s*)?\(", re.MULTILINE)),
        ("function", re.compile(r"^const\s+(\w+)\s*=\s*(?:async\s*)?\(", re.MULTILINE)),
        ("class", re.compile(r"^export\s+class\s+(\w+)", re.MULTILINE)),
        ("class", re.compile(r"^class\s+(\w+)", re.MULTILINE)),
    ],
    "typescript": [
        ("function", re.compile(r"^export\s+default\s+async\s+function\s+(\w+)\s*\(", re.MULTILINE)),
        ("function", re.compile(r"^export\s+default\s+function\s+(\w+)\s*\(", re.MULTILINE)),
        ("function", re.compile(r"^export\s+async\s+function\s+(\w+)\s*\(", re.MULTILINE)),
        ("function", re.compile(r"^export\s+function\s+(\w+)\s*\(", re.MULTILINE)),
        ("function", re.compile(r"^async\s+function\s+(\w+)\s*\(", re.MULTILINE)),
        ("function", re.compile(r"^function\s+(\w+)\s*\(", re.MULTILINE)),
        ("function", re.compile(r"^export\s+const\s+(\w+)\s*=\s*(?:async\s*)?\(", re.MULTILINE)),
        ("function", re.compile(r"^const\s+(\w+)\s*=\s*(?:async\s*)?\(", re.MULTILINE)),
        ("class", re.compile(r"^export\s+class\s+(\w+)", re.MULTILINE)),
        ("class", re.compile(r"^class\s+(\w+)", re.MULTILINE)),
    ],
    "typescriptreact": [
        ("function", re.compile(r"^export\s+default\s+async\s+function\s+(\w+)\s*\(", re.MULTILINE)),
        ("function", re.compile(r"^export\s+default\s+function\s+(\w+)\s*\(", re.MULTILINE)),
        ("function", re.compile(r"^export\s+async\s+function\s+(\w+)\s*\(", re.MULTILINE)),
        ("function", re.compile(r"^export\s+function\s+(\w+)\s*\(", re.MULTILINE)),
        ("function", re.compile(r"^async\s+function\s+(\w+)\s*\(", re.MULTILINE)),
        ("function", re.compile(r"^function\s+(\w+)\s*\(", re.MULTILINE)),
        ("function", re.compile(r"^export\s+const\s+(\w+)\s*=\s*(?:async\s*)?\(", re.MULTILINE)),
        ("function", re.compile(r"^const\s+(\w+)\s*=\s*(?:async\s*)?\(", re.MULTILINE)),
        ("class", re.compile(r"^export\s+class\s+(\w+)", re.MULTILINE)),
        ("class", re.compile(r"^class\s+(\w+)", re.MULTILINE)),
    ],
    "java": [
        ("class", re.compile(r"^class\s+(\w+)", re.MULTILINE)),
        (
            "function",
            re.compile(
                r"^(?:public|private|protected)?\s*(?:static\s+)?[\w<>\[\]]+\s+(\w+)\s*\(",
                re.MULTILINE,
            ),
        ),
    ],
    "cpp": [
        ("class", re.compile(r"^class\s+(\w+)", re.MULTILINE)),
        (
            "function",
            re.compile(
                r"^(?:[\w:<>~*&]+\s+)+(\w+)\s*\(",
                re.MULTILINE,
            ),
        ),
    ],
    "c": [
        (
            "function",
            re.compile(
                r"^\s*(?:[\w*]+\s+)+(\w+)\s*\(",
                re.MULTILINE,
            ),
        ),
    ],
}


def extract_symbol_spans(language: str, text: str) -> list[SymbolSpan]:
    if not text:
        return []

    matches: list[tuple[int, str, str]] = []
    for chunk_type, pattern in PATTERNS.get(language, []):
        for match in pattern.finditer(text):
            matches.append((match.start(), chunk_type, match.group(1)))

    if not matches:
        return []

    matches.sort(key=lambda item: item[0])
    deduped: list[tuple[int, str, str]] = []
    seen_offsets: set[int] = set()
    for offset, chunk_type, symbol_name in matches:
        if offset in seen_offsets:
            continue
        seen_offsets.add(offset)
        deduped.append((offset, chunk_type, symbol_name))

    total_lines = len(text.splitlines())
    spans: list[SymbolSpan] = []
    for index, (offset, chunk_type, symbol_name) in enumerate(deduped):
        start_line = text.count("\n", 0, offset) + 1
        if index + 1 < len(deduped):
            next_offset = deduped[index + 1][0]
            end_line = text.count("\n", 0, next_offset)
        else:
            end_line = total_lines
        spans.append(
            SymbolSpan(
                chunk_type=chunk_type,
                symbol_name=symbol_name,
                start_line=start_line,
                end_line=end_line,
            )
        )
    return spans


def infer_symbol_metadata(language: str, text: str) -> tuple[str, str | None]:
    spans = extract_symbol_spans(language, text)
    if spans:
        return spans[0].chunk_type, spans[0].symbol_name
    return "text_chunk", None

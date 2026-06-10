from __future__ import annotations

from codebase_rag.config import Settings
from codebase_rag.symbols import extract_symbol_spans, infer_symbol_metadata
from codebase_rag.types import Chunk, SourceFile


def _build_chunk(
    source_file: SourceFile,
    lines: list[str],
    start_line: int,
    end_line: int,
    chunk_type: str,
    symbol_name: str | None,
) -> Chunk:
    chunk_text = "\n".join(lines[start_line - 1 : end_line])
    return Chunk(
        chunk_id=f"{source_file.rel_path}:{start_line}-{end_line}",
        file_path=source_file.rel_path,
        language=source_file.language,
        start_line=start_line,
        end_line=end_line,
        content=chunk_text,
        chunk_type=chunk_type,
        symbol_name=symbol_name,
    )


def _line_chunks_for_range(
    source_file: SourceFile,
    lines: list[str],
    settings: Settings,
    start_line: int,
    end_line: int,
    chunk_type: str = "text_chunk",
    symbol_name: str | None = None,
) -> list[Chunk]:
    step = max(1, settings.chunk_size - settings.chunk_overlap)
    chunks: list[Chunk] = []
    start_index = start_line - 1
    stop_index = end_line

    while start_index < stop_index:
        end_index = min(start_index + settings.chunk_size, stop_index)
        chunk_start_line = start_index + 1
        chunk_end_line = end_index
        chunk_text = "\n".join(lines[start_index:end_index])
        if symbol_name is None and chunk_type == "text_chunk":
            inferred_chunk_type, inferred_symbol_name = infer_symbol_metadata(
                source_file.language,
                chunk_text,
            )
        else:
            inferred_chunk_type = chunk_type
            inferred_symbol_name = symbol_name

        chunks.append(
            _build_chunk(
                source_file,
                lines,
                chunk_start_line,
                chunk_end_line,
                inferred_chunk_type,
                inferred_symbol_name,
            )
        )

        if end_index == stop_index:
            break
        start_index += step

    return chunks


def chunk_source_file(source_file: SourceFile, settings: Settings) -> list[Chunk]:
    lines = source_file.content.splitlines()
    if not lines:
        return []

    spans = extract_symbol_spans(source_file.language, source_file.content)
    if not spans:
        return _line_chunks_for_range(source_file, lines, settings, 1, len(lines))

    chunks: list[Chunk] = []
    next_start_line = 1
    for span in spans:
        if next_start_line < span.start_line:
            chunks.extend(
                _line_chunks_for_range(
                    source_file,
                    lines,
                    settings,
                    next_start_line,
                    span.start_line - 1,
                )
            )
        chunks.extend(
            _line_chunks_for_range(
                source_file,
                lines,
                settings,
                span.start_line,
                span.end_line,
                span.chunk_type,
                span.symbol_name,
            )
        )
        next_start_line = span.end_line + 1

    if next_start_line <= len(lines):
        chunks.extend(
            _line_chunks_for_range(
                source_file,
                lines,
                settings,
                next_start_line,
                len(lines),
            )
        )

    return chunks


def render_chunk_document(chunk: Chunk) -> str:
    symbol_name = chunk.symbol_name or "N/A"
    return (
        f"File: {chunk.file_path}\n"
        f"Language: {chunk.language}\n"
        f"Symbol: {symbol_name}\n"
        f"Type: {chunk.chunk_type}\n"
        f"Lines: {chunk.start_line}-{chunk.end_line}\n\n"
        f"Code:\n{chunk.content}"
    )

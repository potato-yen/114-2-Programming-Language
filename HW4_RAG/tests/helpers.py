from codebase_rag.types import Chunk


def make_chunk(file_path: str, start_line: int, end_line: int, content: str) -> Chunk:
    return Chunk(
        chunk_id=f"{file_path}:{start_line}-{end_line}",
        file_path=file_path,
        language="python",
        start_line=start_line,
        end_line=end_line,
        content=content,
    )

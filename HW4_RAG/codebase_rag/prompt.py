from __future__ import annotations


def build_answer_prompt(question: str, retrieved_context: str) -> str:
    return f"""You are a codebase navigation assistant.
Answer the user's question using only the provided retrieved code chunks.
Do not invent files, functions, or line numbers.
If the retrieved chunks are insufficient, say that the result is uncertain.
Return relevant files with line ranges and explain briefly why each one is relevant.

User question:
{question}

Retrieved chunks:
{retrieved_context}
"""

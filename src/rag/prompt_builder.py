from typing import Iterable

SYSTEM_PROMPT = """You are a reliable RAG assistant.
Answer the user's question using only the provided context.
If the answer is not supported by the context, say that the context is insufficient.
Do not make up facts.
Be concise and accurate.
"""

def build_context_block(contexts: Iterable[str]) -> str:
    formatted = []
    for idx, text in enumerate(contexts, start=1):
        formatted.append(f"[Context {idx}]\n{text}")
    return "\n\n".join(formatted)

def build_rag_prompt(question: str, contexts: Iterable[str]) -> str:
    context_block = build_context_block(contexts)

    return f"""{SYSTEM_PROMPT}

            Context:
            {context_block}

            Question:
            {question}

            Answer:
            """
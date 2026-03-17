from time import perf_counter
from src.rag.generator import generate_answer
from src.rag.prompt_builder import build_rag_prompt
from src.rag.retriever import retrieve

def run_rag(query: str, k: int = 3) -> dict:
    start = perf_counter()
    retrieved_docs = retrieve(query, k=k)
    contexts = [doc.page_content for doc in retrieved_docs]
    prompt = build_rag_prompt(query, contexts)
    answer = generate_answer(prompt)

    latency_ms = round((perf_counter() - start) * 1000, 2)

    return {
        "question": query,
        "retrieved_docs": [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in retrieved_docs
        ],
        "prompt": prompt,
        "answer": answer,
        "latency_ms": latency_ms,
    }
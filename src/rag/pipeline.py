from time import perf_counter
from src.rag.generator import generate_answer
from src.rag.prompt_builder import build_rag_prompt
from src.rag.retriever import retrieve

def run_rag(query: str, k: int = 3) -> dict:
    overall_start = perf_counter()
    retrieval_start = perf_counter()
    retrieved_docs = retrieve(query, k=k)
    retrieval_latency_ms = round((perf_counter() - retrieval_start) * 1000, 2)

    contexts = [doc.page_content for doc in retrieved_docs]
    prompt = build_rag_prompt(query, contexts)

    generation_start = perf_counter()
    answer = generate_answer(prompt)
    generation_latency_ms = round((perf_counter() - generation_start) * 1000, 2)

    total_latency_ms = round((perf_counter() - overall_start) * 1000, 2)

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
        "latency_ms": total_latency_ms,
        "retrieval_latency_ms": retrieval_latency_ms,
        "generation_latency_ms": generation_latency_ms,
    }
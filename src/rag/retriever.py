from typing import Any
from src.indexing.chroma_store import get_vector_store

def retrieve(
    query: str,
    k: int = 3,
    collection_name: str = "llm_reliability_lab",
) -> list[Any]:
    vector_store = get_vector_store(collection_name=collection_name)
    return vector_store.similarity_search(query, k=k)
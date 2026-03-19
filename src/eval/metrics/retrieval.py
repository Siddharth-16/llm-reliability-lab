def recall_at_k(reference_doc_ids: list[str], retrieved_doc_ids: list[str]) -> float:
    if not reference_doc_ids:
        return 0.0

    hits = sum(1 for doc_id in reference_doc_ids if doc_id in retrieved_doc_ids)
    return hits / len(reference_doc_ids)
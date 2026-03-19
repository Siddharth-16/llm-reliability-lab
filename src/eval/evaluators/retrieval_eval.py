from src.eval.metrics.retrieval import recall_at_k

def evaluate_retrieval(sample, retrieved_docs) -> dict:
    retrieved_ids = [doc["metadata"]["doc_id"] for doc in retrieved_docs]

    recall = recall_at_k(
        reference_doc_ids=sample.reference_doc_ids or [],
        retrieved_doc_ids=retrieved_ids,
    )

    return {
        "retrieval_recall_at_k": recall,
    }
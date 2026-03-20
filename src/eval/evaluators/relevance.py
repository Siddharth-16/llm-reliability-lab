from src.eval.judges.llm_judge import judge_with_llm, parse_json_response

def build_context_relevance_prompt(question: str, contexts: list[str]) -> str:
    context_block = "\n\n".join(
        [f"[Context {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)]
    )

    return f"""
You are evaluating whether the retrieved context is relevant to the user's question.

Return ONLY valid JSON in this format:
{{
  "score": 0.0 to 1.0,
  "label": "relevant" or "partially_relevant" or "irrelevant",
  "reasoning": "short explanation"
}}

Question:
{question}

Retrieved Context:
{context_block}

Evaluate how relevant the retrieved context is for answering the question.
Return JSON only.
""".strip()

def evaluate_context_relevance(sample, retrieved_docs: list[dict]) -> dict:
    reference_ids = set(sample.reference_doc_ids or [])
    retrieved_ids = [doc["metadata"]["doc_id"] for doc in retrieved_docs]

    if not reference_ids:
        return {
            "context_relevance": 0.0,
            "context_relevance_label": "unknown",
            "context_relevance_reasoning": "No reference_doc_ids available for this sample.",
        }

    if retrieved_ids and retrieved_ids[0] in reference_ids:
        return {
            "context_relevance": 1.0,
            "context_relevance_label": "relevant",
            "context_relevance_reasoning": "Top retrieved document matches a reference document.",
        }

    if any(doc_id in reference_ids for doc_id in retrieved_ids):
        return {
            "context_relevance": 0.5,
            "context_relevance_label": "partially_relevant",
            "context_relevance_reasoning": "A reference document appears in retrieved results, but not at rank 1.",
        }

    return {
        "context_relevance": 0.0,
        "context_relevance_label": "irrelevant",
        "context_relevance_reasoning": "No retrieved documents match the reference documents.",
    }
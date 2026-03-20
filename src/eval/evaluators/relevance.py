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

def evaluate_context_relevance(question: str, retrieved_docs: list[dict]) -> dict:
    contexts = [doc["content"] for doc in retrieved_docs]
    prompt = build_context_relevance_prompt(question, contexts)
    raw_response = judge_with_llm(prompt)
    parsed = parse_json_response(raw_response)

    return {
        "context_relevance": parsed.get("score", 0.0),
        "context_relevance_label": parsed.get("label", "unknown"),
        "context_relevance_reasoning": parsed.get("reasoning", ""),
    }
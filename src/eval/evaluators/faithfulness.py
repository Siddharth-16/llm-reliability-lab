from src.eval.judges.llm_judge import judge_with_llm, parse_json_response

def build_faithfulness_prompt(question: str, answer: str, contexts: list[str]) -> str:
    context_block = "\n\n".join(
        [f"[Context {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)]
    )

    return f"""
You are evaluating whether an answer is faithful to the provided context.

Return ONLY valid JSON in this format:
{{
  "score": 0.0 to 1.0,
  "label": "faithful" or "partially_faithful" or "unfaithful",
  "reasoning": "short explanation"
}}

Question:
{question}

Retrieved Context:
{context_block}

Answer:
{answer}

Evaluate whether the answer is fully supported by the retrieved context.
If the answer includes unsupported claims, reduce the score.
Return JSON only.
""".strip()

def evaluate_faithfulness(question: str, answer: str, retrieved_docs: list[dict]) -> dict:
    contexts = [doc["content"] for doc in retrieved_docs]
    prompt = build_faithfulness_prompt(question, answer, contexts)
    raw_response = judge_with_llm(prompt)
    parsed = parse_json_response(raw_response)

    return {
        "faithfulness": parsed.get("score", 0.0),
        "faithfulness_label": parsed.get("label", "unknown"),
        "faithfulness_reasoning": parsed.get("reasoning", ""),
    }
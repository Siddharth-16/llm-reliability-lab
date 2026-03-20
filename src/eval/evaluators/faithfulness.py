from src.eval.judges.llm_judge import judge_with_llm, parse_json_response

def build_faithfulness_prompt(question: str, answer: str, contexts: list[str]) -> str:
    context_block = "\n\n".join(
        [f"[Context {i + 1}]\n{ctx}" for i, ctx in enumerate(contexts)]
    )

    return f"""
You are a strict evaluator of answer faithfulness.

Your task:
Decide whether the answer is supported by the retrieved context.

Important rules:
- Ignore minor wording differences.
- Do not require exact wording.
- Judge only whether the answer is supported by the context.
- Use only these scores:
  - 1.0 = fully supported
  - 0.5 = partially supported
  - 0.0 = unsupported
- Return only valid JSON.
- Do not add markdown or explanation outside JSON.

Return JSON in exactly this format:
{{
  "score": 1.0,
  "label": "faithful",
  "reasoning": "short explanation"
}}

Allowed labels:
- "faithful"
- "partially_faithful"
- "unfaithful"

Question:
{question}

Retrieved Context:
{context_block}

Answer:
{answer}
""".strip()

def evaluate_faithfulness(question: str, answer: str, retrieved_docs: list[dict]) -> dict:
    contexts = [doc["content"] for doc in retrieved_docs[:2]]
    prompt = build_faithfulness_prompt(question, answer, contexts)
    raw_response = judge_with_llm(prompt)
    parsed = parse_json_response(raw_response)

    score = parsed.get("score", 0.0)
    if score not in {0.0, 0.5, 1.0}:
        score = 0.0

    return {
        "faithfulness": score,
        "faithfulness_label": parsed.get("label", "unknown"),
        "faithfulness_reasoning": parsed.get("reasoning", ""),
    }
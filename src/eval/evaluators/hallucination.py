from src.eval.judges.llm_judge import judge_with_llm, parse_json_response

def build_hallucination_prompt(question: str, answer: str, contexts: list[str]) -> str:
    context_block = "\n\n".join(
        [f"[Context {i + 1}]\n{ctx}" for i, ctx in enumerate(contexts)]
    )

    return f"""
You are a strict evaluator of hallucination in RAG answers.

Your task:
Decide whether the answer contains information that is not supported by the retrieved context.

Important rules:
- Ignore minor wording differences.
- If the answer is fully supported, hallucination score is 0.0.
- Use only these scores:
  - 0.0 = no hallucination
  - 0.5 = possible or partial hallucination
  - 1.0 = clear hallucination
- Return only valid JSON.
- Do not add markdown or explanation outside JSON.

Return JSON in exactly this format:
{{
  "score": 0.0,
  "label": "no_hallucination",
  "reasoning": "short explanation"
}}

Allowed labels:
- "no_hallucination"
- "possible_hallucination"
- "hallucination"

Question:
{question}

Retrieved Context:
{context_block}

Answer:
{answer}
""".strip()

def evaluate_hallucination(question: str, answer: str, retrieved_docs: list[dict]) -> dict:
    contexts = [doc["content"] for doc in retrieved_docs[:2]]
    prompt = build_hallucination_prompt(question, answer, contexts)
    raw_response = judge_with_llm(prompt)
    parsed = parse_json_response(raw_response)

    score = parsed.get("score", 0.0)
    if score not in {0.0, 0.5, 1.0}:
        score = 0.0

    return {
        "hallucination_score": score,
        "hallucination_label": parsed.get("label", "unknown"),
        "hallucination_reasoning": parsed.get("reasoning", ""),
    }
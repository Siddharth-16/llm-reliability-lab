from src.eval.judges.llm_judge import judge_with_llm, parse_json_response

def build_hallucination_prompt(question: str, answer: str, contexts: list[str]) -> str:
    context_block = "\n\n".join(
        [f"[Context {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)]
    )

    return f"""
You are evaluating whether an answer contains hallucinated or unsupported information.

Return ONLY valid JSON in this format:
{{
  "score": 0.0 to 1.0,
  "label": "no_hallucination" or "possible_hallucination" or "hallucination",
  "reasoning": "short explanation"
}}

Scoring rule:
- 0.0 means no hallucination
- 1.0 means severe hallucination

Question:
{question}

Retrieved Context:
{context_block}

Answer:
{answer}

Determine whether the answer contains information that is not supported by the context.
Return JSON only.
""".strip()

def evaluate_hallucination(question: str, answer: str, retrieved_docs: list[dict]) -> dict:
    contexts = [doc["content"] for doc in retrieved_docs]
    prompt = build_hallucination_prompt(question, answer, contexts)
    raw_response = judge_with_llm(prompt)
    parsed = parse_json_response(raw_response)

    return {
        "hallucination_score": parsed.get("score", 0.0),
        "hallucination_label": parsed.get("label", "unknown"),
        "hallucination_reasoning": parsed.get("reasoning", ""),
    }
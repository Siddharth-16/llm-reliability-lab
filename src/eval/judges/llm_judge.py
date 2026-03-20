import json
from typing import Any
from src.models.llm_factory import get_llm

def judge_with_llm(prompt: str) -> str:
    llm = get_llm()
    return llm.generate(prompt)

def parse_json_response(response: str) -> dict[str, Any]:
    response = response.strip()

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        start = response.find("{")
        end = response.rfind("}")

        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(response[start : end + 1])
            except json.JSONDecodeError:
                pass

    return {
        "score": 0.0,
        "label": "parse_error",
        "reasoning": response,
    }
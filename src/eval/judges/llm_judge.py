import json
from typing import Any
from src.models.llm_factory import get_llm

def judge_with_llm(prompt: str) -> str:
    llm = get_llm()
    return llm.generate(prompt)

def parse_json_response(response: str) -> dict[str, Any]:
    response = response.strip()

    try:
        parsed = json.loads(response)
        return parsed if isinstance(parsed, dict) else _default_parse_error(response)
    except json.JSONDecodeError:
        pass

    start = response.find("{")
    end = response.rfind("}")

    if start != -1 and end != -1 and end > start:
        candidate = response[start : end + 1]
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else _default_parse_error(response)
        except json.JSONDecodeError:
            pass

    return _default_parse_error(response)


def _default_parse_error(response: str) -> dict[str, Any]:
    return {
        "score": 0.0,
        "label": "parse_error",
        "reasoning": response,
    }
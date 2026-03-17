import json
from urllib import request
from src.config.settings import settings

class MockLLM:
    def generate(self, prompt: str) -> str:
        prompt_lower = prompt.lower()

        if "multi-factor authentication" in prompt_lower or "mfa" in prompt_lower:
            return (
                "Yes. The context says that all employees must use "
                "multi-factor authentication for company systems."
            )

        if "reimbursable" in prompt_lower or "$75" in prompt_lower:
            return "Meals are reimbursable up to $75 per day when traveling for work."

        return "The context is insufficient to answer the question."

class OllamaLLM:
    def __init__(self, model: str, base_url: str) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }

        req = request.Request(
            url=f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with request.urlopen(req) as response:
            body = json.loads(response.read().decode("utf-8"))

        return body.get("response", "").strip()

def get_llm():
    provider = settings.llm_provider.lower()

    if provider == "ollama":
        return OllamaLLM(
            model=settings.llm_model,
            base_url=settings.ollama_base_url,
        )

    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
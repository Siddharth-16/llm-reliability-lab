from typing import Protocol

class LLMAdapter(Protocol):
    def generate(self, prompt: str) -> str:
        """Generate a response from the given prompt."""
        ...
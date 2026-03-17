from src.models.llm_factory import get_llm

def generate_answer(prompt: str) -> str:
    llm = get_llm()
    return llm.generate(prompt)
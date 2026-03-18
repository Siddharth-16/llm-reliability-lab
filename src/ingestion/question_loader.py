import json
from typing import List
from src.ingestion.schemas import QuestionSample

def load_questions(path: str) -> List[QuestionSample]:
    questions = []

    with open(path, "r") as f:
        for line in f:
            data = json.loads(line)
            questions.append(QuestionSample(**data))

    return questions
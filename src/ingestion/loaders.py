import json
from pathlib import Path
from typing import List

from src.ingestion.schemas import Document


def load_documents(path: str) -> List[Document]:
    documents = []

    with open(path, "r") as f:
        for line in f:
            data = json.loads(line)
            documents.append(Document(**data))

    return documents
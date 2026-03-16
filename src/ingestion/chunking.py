import uuid
from typing import List

from src.ingestion.schemas import Chunk, Document

def chunk_document(
    document: Document,
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[Chunk]:

    text = document.text
    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size
        chunk_text = text[start:end]

        chunk = Chunk(
            chunk_id=str(uuid.uuid4()),
            doc_id=document.doc_id,
            text=chunk_text,
            metadata={
                "title": document.title,
                "source": document.source,
            },
        )

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks
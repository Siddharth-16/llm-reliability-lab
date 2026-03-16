from typing import List
from src.ingestion.schemas import Document, Chunk
from src.ingestion.chunking import chunk_document

def process_documents(documents: List[Document]) -> List[Chunk]:

    all_chunks = []

    for doc in documents:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)

    return all_chunks
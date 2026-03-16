from typing import Iterable
from langchain_core.documents import Document as LangChainDocument
from langchain_chroma import Chroma
from src.config.settings import settings
from src.indexing.embedder import get_embedding_model

def get_vector_store(collection_name: str = "llm_reliability_lab") -> Chroma:
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embedding_model(),
        persist_directory=settings.vector_db_dir,
    )

def add_documents_to_store(
    documents: Iterable[LangChainDocument],
    collection_name: str = "llm_reliability_lab",
) -> Chroma:
    vector_store = get_vector_store(collection_name=collection_name)
    vector_store.add_documents(list(documents))
    return vector_store
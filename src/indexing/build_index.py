from langchain_core.documents import Document as LangChainDocument
from src.ingestion.loaders import load_documents
from src.ingestion.pipeline import process_documents
from src.indexing.chroma_store import add_documents_to_store

def build_index(
    documents_path: str = "datasets/sample_corpus/documents.jsonl",
    collection_name: str = "llm_reliability_lab",
) -> None:
    documents = load_documents(documents_path)
    chunks = process_documents(documents)

    langchain_docs = []
    for chunk in chunks:
        langchain_docs.append(
            LangChainDocument(
                page_content=chunk.text,
                metadata={
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    **chunk.metadata,
                },
            )
        )

    add_documents_to_store(langchain_docs, collection_name=collection_name)
    print(f"Indexed {len(documents)} documents")
    print(f"Stored {len(chunks)} chunks in Chroma")
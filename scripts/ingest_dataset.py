from src.ingestion.loaders import load_documents
from src.ingestion.pipeline import process_documents

def main():
    docs = load_documents("datasets/sample_corpus/documents.jsonl")
    chunks = process_documents(docs)

    print(f"Loaded documents: {len(docs)}")
    print(f"Generated chunks: {len(chunks)}")

    print("\nExample chunk:\n")
    print(chunks[0])


if __name__ == "__main__":
    main()
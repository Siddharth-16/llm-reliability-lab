from src.rag.retriever import retrieve

def main() -> None:
    query = "Is MFA required for company systems?"
    results = retrieve(query, k=2)

    print(f"Query: {query}\n")
    print(f"Retrieved {len(results)} results\n")

    for i, doc in enumerate(results, start=1):
        print(f"Result {i}")
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")
        print("-" * 80)

if __name__ == "__main__":
    main()

from src.rag.pipeline import run_rag

def main() -> None:
    query = "Is MFA required for company systems?"
    result = run_rag(query, k=2)

    print(f"Question: {result['question']}\n")
    print("Retrieved Contexts:\n")

    for idx, doc in enumerate(result["retrieved_docs"], start=1):
        print(f"Context {idx}:")
        print(doc["content"])
        print(doc["metadata"])
        print("-" * 80)

    print("\nGenerated Answer:\n")
    print(result["answer"])

    print(f"\nLatency: {result['latency_ms']} ms")

if __name__ == "__main__":
    main()
from time import perf_counter
from src.eval.evaluators.answer_eval import evaluate_answer
from src.eval.evaluators.faithfulness import evaluate_faithfulness
from src.eval.evaluators.hallucination import evaluate_hallucination
from src.eval.evaluators.relevance import evaluate_context_relevance
from src.eval.evaluators.retrieval_eval import evaluate_retrieval
from src.ingestion.question_loader import load_questions
from src.rag.pipeline import run_rag

def run_evaluation(dataset_path: str) -> dict:
    samples = load_questions(dataset_path)
    results = []

    start_time = perf_counter()

    for sample in samples:
        rag_output = run_rag(sample.question)

        retrieval_metrics = evaluate_retrieval(sample, rag_output["retrieved_docs"])
        answer_metrics = evaluate_answer(sample, rag_output["answer"])
        faithfulness_metrics = evaluate_faithfulness(
            sample.question,
            rag_output["answer"],
            rag_output["retrieved_docs"],
        )
        hallucination_metrics = evaluate_hallucination(
            sample.question,
            rag_output["answer"],
            rag_output["retrieved_docs"],
        )
        relevance_metrics = evaluate_context_relevance(
            sample,
            rag_output["retrieved_docs"],
        )

        result = {
            "sample_id": sample.sample_id,
            "question": sample.question,
            "ground_truth": sample.ground_truth_answer,
            "prediction": rag_output["answer"],
            "retrieved_docs": rag_output["retrieved_docs"],
            "latency_ms": rag_output["latency_ms"],
            "retrieval_latency_ms": rag_output["retrieval_latency_ms"],
            "generation_latency_ms": rag_output["generation_latency_ms"],
            "metrics": {
                **retrieval_metrics,
                **answer_metrics,
                **faithfulness_metrics,
                **hallucination_metrics,
                **relevance_metrics,
            },
        }

        results.append(result)

    total_time = perf_counter() - start_time

    summary = {
        "num_samples": len(results),
        "avg_recall": sum(r["metrics"]["retrieval_recall_at_k"] for r in results) / len(results),
        "avg_exact_match": sum(r["metrics"].get("exact_match", 0.0) for r in results) / len(results),
        "avg_token_f1": sum(r["metrics"].get("token_f1", 0.0) for r in results) / len(results),
        "avg_faithfulness": sum(r["metrics"].get("faithfulness", 0.0) for r in results) / len(results),
        "avg_hallucination_score": sum(
            r["metrics"].get("hallucination_score", 0.0) for r in results
        ) / len(results),
        "avg_context_relevance": sum(
            r["metrics"].get("context_relevance", 0.0) for r in results
        ) / len(results),
        "avg_latency_ms": sum(r["latency_ms"] for r in results) / len(results),
        "avg_retrieval_latency_ms": sum(
            r["retrieval_latency_ms"] for r in results
        ) / len(results),
        "avg_generation_latency_ms": sum(
            r["generation_latency_ms"] for r in results
        ) / len(results),
        "latency_sec": round(total_time, 2),
    }

    return {
        "results": results,
        "summary": summary,
    }
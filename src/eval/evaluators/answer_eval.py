from src.eval.metrics.generation import exact_match

def evaluate_answer(sample, prediction: str) -> dict:
    if not sample.ground_truth_answer:
        return {}

    score = exact_match(prediction, sample.ground_truth_answer)

    return {
        "exact_match": score,
    }
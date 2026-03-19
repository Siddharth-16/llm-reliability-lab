from src.eval.metrics.generation import exact_match, token_f1

def evaluate_answer(sample, prediction: str) -> dict:
    if not sample.ground_truth_answer:
        return {}

    return {
        "exact_match": exact_match(prediction, sample.ground_truth_answer),
        "token_f1": token_f1(prediction, sample.ground_truth_answer),
    }
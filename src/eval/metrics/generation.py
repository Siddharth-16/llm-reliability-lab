def exact_match(predicted: str, ground_truth: str) -> float:
    if not predicted or not ground_truth:
        return 0.0

    return float(predicted.strip().lower() == ground_truth.strip().lower())
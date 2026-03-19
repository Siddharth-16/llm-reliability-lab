import re

def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())

def exact_match(predicted: str, ground_truth: str) -> float:
    if not predicted or not ground_truth:
        return 0.0

    return float(normalize_text(predicted) == normalize_text(ground_truth))

def token_f1(predicted: str, ground_truth: str) -> float:
    pred_tokens = normalize_text(predicted).split()
    gt_tokens = normalize_text(ground_truth).split()

    if not pred_tokens or not gt_tokens:
        return 0.0

    common = set(pred_tokens) & set(gt_tokens)
    num_same = sum(min(pred_tokens.count(tok), gt_tokens.count(tok)) for tok in common)

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)

    return 2 * precision * recall / (precision + recall)
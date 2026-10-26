from typing import Dict, Iterable, List


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def classification_metrics(gold: Iterable[int], predicted: Iterable[int]) -> Dict[str, object]:
    gold_list: List[int] = list(gold)
    predicted_list: List[int] = list(predicted)
    if len(gold_list) != len(predicted_list):
        raise ValueError("Gold and predicted label counts differ")
    if not gold_list:
        raise ValueError("Cannot score an empty dataset")

    tp = sum(g == 1 and p == 1 for g, p in zip(gold_list, predicted_list))
    tn = sum(g == 0 and p == 0 for g, p in zip(gold_list, predicted_list))
    fp = sum(g == 0 and p == 1 for g, p in zip(gold_list, predicted_list))
    fn = sum(g == 1 and p == 0 for g, p in zip(gold_list, predicted_list))

    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall)

    negative_precision = _safe_divide(tn, tn + fn)
    negative_recall = _safe_divide(tn, tn + fp)
    negative_f1 = _safe_divide(
        2 * negative_precision * negative_recall,
        negative_precision + negative_recall,
    )

    return {
        "accuracy": _safe_divide(tp + tn, len(gold_list)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": (f1 + negative_f1) / 2,
        "support": len(gold_list),
        "confusion_matrix": {
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
        },
    }

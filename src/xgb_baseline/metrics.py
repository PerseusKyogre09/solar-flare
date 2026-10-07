from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, matthews_corrcoef, roc_auc_score


def safe_divide(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def binary_metrics(y_true, probability, threshold: float = 0.5) -> dict:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    pred = p >= threshold
    tp = int(np.sum(pred & (y == 1))); tn = int(np.sum(~pred & (y == 0)))
    fp = int(np.sum(pred & (y == 0))); fn = int(np.sum(~pred & (y == 1)))
    precision = safe_divide(tp, tp + fp); recall = safe_divide(tp, tp + fn)
    hss = safe_divide(2 * (tp * tn - fn * fp), (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    false_positive_rate = safe_divide(fp, fp + tn)
    true_negative_rate = safe_divide(tn, tn + fp)
    out = {"threshold": float(threshold), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
           "accuracy": safe_divide(tp + tn, tp + tn + fp + fn), "pod_recall_tpr": recall,
           "far": safe_divide(fp, tp + fp), "csi": safe_divide(tp, tp + fn + fp),
           "hss": hss, "tss": recall - false_positive_rate, "false_positive_rate": false_positive_rate,
           "true_negative_rate": true_negative_rate, "specificity": true_negative_rate, "precision": precision,
           "f1": safe_divide(2 * precision * recall, precision + recall), "mcc": float(matthews_corrcoef(y, pred)) if np.unique(y).size > 1 else 0.0,
           "brier_score": float(brier_score_loss(y, p))}
    if np.unique(y).size < 2:
        out["roc_auc"] = float("nan")
    else:
        out["roc_auc"] = float(roc_auc_score(y, p))
    if np.sum(y == 1) == 0:
        out["pr_auc"] = float("nan")
    else:
        out["pr_auc"] = float(average_precision_score(y, p))
    return out


def comprehensive_binary_metrics(y_true, probability, threshold: float) -> dict:
    """Complete publication-facing binary test evaluation."""
    y = np.asarray(y_true, dtype=int); p = np.asarray(probability, dtype=float)
    out = binary_metrics(y, p, threshold)
    out.update({"positive_prevalence": float(y.mean()), "negative_count": int((y == 0).sum()),
                "positive_count": int((y == 1).sum()), "total_test_samples": int(y.size),
                "confusion_matrix": confusion_matrix(y, p >= threshold, labels=[0, 1]).tolist(),
                "normalized_confusion_matrix": (confusion_matrix(y, p >= threshold, labels=[0, 1]).astype(float) /
                    np.maximum(confusion_matrix(y, p >= threshold, labels=[0, 1]).sum(axis=1, keepdims=True), 1)).tolist()})
    return out


def threshold_table(y_true, probability, thresholds: list[float]) -> list[dict]:
    return [binary_metrics(y_true, probability, t) for t in thresholds]


def select_validation_threshold(y_true, probability, thresholds: list[float]) -> float:
    rows = threshold_table(y_true, probability, thresholds)
    valid = [r for r in rows if np.isfinite(r["tss"])]
    return float(max(valid, key=lambda r: (r["tss"], r["pod_recall_tpr"]))["threshold"]) if valid else 0.5

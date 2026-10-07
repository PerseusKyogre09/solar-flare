from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


BINARY_BOOTSTRAP_METRICS = (
    "roc_auc",
    "pr_auc",
    "tss",
    "hss",
    "csi",
    "mcc",
    "f1",
    "precision",
    "pod_recall_tpr",
    "specificity",
    "far",
    "brier_score",
)


def expected_calibration_error(y_true, probability, n_bins: int = 10) -> float:
    """Compute ECE using fixed-width bins over [0, 1]."""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(probability, dtype=np.float64)
    if y.ndim != 1 or p.shape != y.shape or not y.size:
        raise ValueError("y_true and probability must be aligned non-empty vectors")
    if n_bins < 1 or np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("n_bins must be positive and probabilities must be finite in [0, 1]")
    if np.any((y != 0) & (y != 1)):
        raise ValueError("y_true must contain only 0 and 1")
    bin_ids = np.minimum((p * n_bins).astype(int), n_bins - 1)
    return float(sum(
        np.mean(bin_ids == index) * abs(y[bin_ids == index].mean() - p[bin_ids == index].mean())
        for index in range(n_bins) if np.any(bin_ids == index)
    ))


def binary_curve_data(y_true, probability, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Return coordinates calculated from actual probabilities for ROC or PR."""
    y = np.asarray(y_true, dtype=np.int8)
    p = np.asarray(probability, dtype=np.float64)
    if y.ndim != 1 or p.shape != y.shape or not y.size:
        raise ValueError("y_true and probability must be aligned non-empty vectors")
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("probability must be finite and in [0, 1]")
    if np.any((y != 0) & (y != 1)) or np.unique(y).size != 2:
        raise ValueError("curve generation requires both binary classes")
    if kind == "roc":
        x, value, _ = roc_curve(y, p)
        return x, value
    if kind == "pr":
        precision, recall, _ = precision_recall_curve(y, p)
        return recall, precision
    raise ValueError("kind must be 'roc' or 'pr'")


def map_feature_groups(feature_names) -> dict[str, str] | None:
    """Map explicit physical feature names; refuse anonymous feature indices."""
    names = [str(name).lower() for name in feature_names]
    if not names or all(name.startswith("feature_") for name in names):
        return None
    groups = {}
    for name in names:
        if any(token in name for token in ("gradient", "grad", "shear")):
            groups[name] = "magnetic_gradient"
        elif any(token in name for token in ("pil", "neutral_line", "neutral-line")):
            groups[name] = "neutral_line_pil"
        elif any(token in name for token in ("wavelet", "wav_")):
            groups[name] = "wavelet"
        elif any(token in name for token in ("flux", "unsigned_flux")):
            groups[name] = "flux"
        else:
            return None
    return groups


def make_cluster_bootstrap_counts(groups, n_resamples: int = 2000, seed: int = 42):
    """Return row-to-cluster codes and cluster multiplicities per resample."""
    group_values = np.asarray(groups)
    if group_values.ndim != 1 or group_values.size == 0:
        raise ValueError("groups must be a non-empty one-dimensional array")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    unique_groups, row_codes = np.unique(group_values, return_inverse=True)
    rng = np.random.default_rng(seed)
    multiplicities = rng.multinomial(
        len(unique_groups),
        np.full(len(unique_groups), 1 / len(unique_groups)),
        size=n_resamples,
    ).astype(np.uint16)
    return unique_groups, row_codes, multiplicities


def cluster_bootstrap_metric_samples(
    y_true,
    probability,
    row_codes,
    cluster_multiplicities,
    threshold: float,
) -> dict[str, np.ndarray]:
    """Compute binary metric replicates using active-region bootstrap weights.

    Scores are sorted once. Weighted ROC-AUC and average precision are then
    evaluated at score-tie groups, while threshold metrics use weighted counts.
    """
    y = np.asarray(y_true, dtype=np.int8)
    p = np.asarray(probability, dtype=np.float64)
    codes = np.asarray(row_codes, dtype=np.int64)
    draws = np.asarray(cluster_multiplicities)
    if y.ndim != 1 or p.shape != y.shape or codes.shape != y.shape:
        raise ValueError("y_true, probability, and row_codes must be aligned vectors")
    if np.any((y != 0) & (y != 1)):
        raise ValueError("y_true must contain only 0 and 1")
    if np.any(~np.isfinite(p)):
        raise ValueError("probability must contain only finite values")
    if draws.ndim != 2 or draws.shape[1] <= int(codes.max(initial=-1)):
        raise ValueError("cluster_multiplicities must have one column per cluster code")

    order = np.argsort(-p, kind="mergesort")
    sorted_probability = p[order]
    starts = np.flatnonzero(
        np.r_[True, sorted_probability[1:] != sorted_probability[:-1]]
    )
    sorted_y = y[order]
    sorted_codes = codes[order]
    result = {name: np.full(len(draws), np.nan, dtype=np.float64) for name in BINARY_BOOTSTRAP_METRICS}
    predicted = p >= threshold
    brier_error = (y - p) ** 2

    for index, cluster_weights in enumerate(draws):
        row_weights = cluster_weights[codes].astype(np.float64, copy=False)
        tp = float(np.sum(row_weights * predicted * y))
        fp = float(np.sum(row_weights * predicted * (1 - y)))
        fn = float(np.sum(row_weights * (1 - predicted) * y))
        tn = float(np.sum(row_weights * (1 - predicted) * (1 - y)))
        total = tp + fp + fn + tn
        positives = tp + fn
        negatives = tn + fp
        predicted_positive = tp + fp
        precision = tp / predicted_positive if predicted_positive else 0.0
        recall = tp / positives if positives else 0.0
        specificity = tn / negatives if negatives else 0.0
        far = fp / predicted_positive if predicted_positive else 0.0
        csi_denominator = tp + fp + fn
        csi = tp / csi_denominator if csi_denominator else 0.0
        hss_denominator = positives * (fn + tn) + predicted_positive * (fp + tn)
        hss = 2 * (tp * tn - fn * fp) / hss_denominator if hss_denominator else 0.0
        mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = (tp * tn - fp * fn) / mcc_denominator if mcc_denominator else 0.0
        result["tss"][index] = recall + specificity - 1.0
        result["hss"][index] = hss
        result["csi"][index] = csi
        result["mcc"][index] = mcc
        result["f1"][index] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result["precision"][index] = precision
        result["pod_recall_tpr"][index] = recall
        result["specificity"][index] = specificity
        result["far"][index] = far
        result["brier_score"][index] = float(np.dot(row_weights, brier_error) / total) if total else np.nan

        sorted_weights = cluster_weights[sorted_codes].astype(np.float64, copy=False)
        group_positive = np.add.reduceat(sorted_weights * sorted_y, starts)
        group_total = np.add.reduceat(sorted_weights, starts)
        group_negative = group_total - group_positive
        nonempty = group_total > 0
        group_positive = group_positive[nonempty]
        group_negative = group_negative[nonempty]
        group_total = group_total[nonempty]
        positive_total = float(group_positive.sum())
        negative_total = float(group_negative.sum())
        if positive_total and negative_total:
            cumulative_negative = np.cumsum(group_negative)
            negative_below = negative_total - cumulative_negative
            result["roc_auc"][index] = float(
                np.sum(group_positive * (negative_below + 0.5 * group_negative))
                / (positive_total * negative_total)
            )
            cumulative_positive = np.cumsum(group_positive)
            cumulative_total = np.cumsum(group_total)
            result["pr_auc"][index] = float(
                np.sum(
                    (group_positive / positive_total)
                    * (cumulative_positive / cumulative_total)
                )
            )

    return result


def percentile_intervals(samples: dict[str, np.ndarray]) -> dict[str, list[float | None]]:
    """Return finite 95% percentile intervals for metric samples."""
    intervals = {}
    for name, values in samples.items():
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        intervals[name] = (
            [float(x) for x in np.percentile(finite, [2.5, 97.5])]
            if finite.size
            else [None, None]
        )
    return intervals


def multiclass_cluster_bootstrap_metric_samples(
    y_true,
    predicted_label,
    row_codes,
    cluster_multiplicities,
    labels=(0, 1, 2),
) -> dict[str, np.ndarray]:
    """Bootstrap macro classification metrics from region-weighted confusion matrices."""
    y = np.asarray(y_true, dtype=np.int64)
    prediction = np.asarray(predicted_label, dtype=np.int64)
    codes = np.asarray(row_codes, dtype=np.int64)
    draws = np.asarray(cluster_multiplicities)
    label_values = np.asarray(labels, dtype=np.int64)
    if y.ndim != 1 or prediction.shape != y.shape or codes.shape != y.shape or not y.size:
        raise ValueError("multiclass inputs must be aligned non-empty vectors")
    if draws.ndim != 2 or draws.shape[1] <= int(codes.max(initial=-1)):
        raise ValueError("cluster_multiplicities must have one column per cluster code")
    if not np.isin(y, label_values).all() or not np.isin(prediction, label_values).all():
        raise ValueError("labels and predictions must belong to labels")

    class_count = len(label_values)
    encoded = y * class_count + prediction
    metrics = ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "macro_tss", "macro_hss", "macro_csi", "macro_far")
    result = {metric: np.full(len(draws), np.nan, dtype=np.float64) for metric in metrics}
    for index, cluster_weights in enumerate(draws):
        row_weights = cluster_weights[codes]
        matrix = np.bincount(encoded, weights=row_weights, minlength=class_count ** 2).reshape(class_count, class_count)
        total = matrix.sum()
        if not total:
            continue
        true_counts = matrix.sum(axis=1)
        predicted_counts = matrix.sum(axis=0)
        tp = np.diag(matrix)
        fn = true_counts - tp
        fp = predicted_counts - tp
        tn = total - tp - fn - fp
        precision = np.divide(tp, predicted_counts, out=np.zeros_like(tp), where=predicted_counts > 0)
        recall = np.divide(tp, true_counts, out=np.zeros_like(tp), where=true_counts > 0)
        specificity = np.divide(tn, tn + fp, out=np.zeros_like(tn), where=(tn + fp) > 0)
        f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
        hss_denominator = true_counts * (fn + tn) + predicted_counts * (fp + tn)
        hss = np.divide(2 * (tp * tn - fn * fp), hss_denominator, out=np.zeros_like(tp), where=hss_denominator > 0)
        csi_denominator = tp + fn + fp
        csi = np.divide(tp, csi_denominator, out=np.zeros_like(tp), where=csi_denominator > 0)
        far = np.divide(fp, predicted_counts, out=np.zeros_like(tp), where=predicted_counts > 0)
        result["accuracy"][index] = float(tp.sum() / total)
        result["macro_precision"][index] = float(precision.mean())
        result["macro_recall"][index] = float(recall.mean())
        result["macro_f1"][index] = float(f1.mean())
        result["weighted_f1"][index] = float(np.average(f1, weights=true_counts))
        result["macro_tss"][index] = float(np.mean(recall + specificity - 1))
        result["macro_hss"][index] = float(hss.mean())
        result["macro_csi"][index] = float(csi.mean())
        result["macro_far"][index] = float(far.mean())
    return result


def sklearn_weighted_rank_metrics(y_true, probability, sample_weight) -> tuple[float, float]:
    """Small reference helper used by tests to compare weighted rank metrics."""
    return (
        float(roc_auc_score(y_true, probability, sample_weight=sample_weight)),
        float(average_precision_score(y_true, probability, sample_weight=sample_weight)),
    )
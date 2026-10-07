"""Create comparison tables and figures from canonical saved predictions."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, matthews_corrcoef

from src.research_metrics import (
    cluster_bootstrap_metric_samples,
    make_cluster_bootstrap_counts,
    multiclass_cluster_bootstrap_metric_samples,
    percentile_intervals,
)
from src.research_metrics import binary_curve_data, expected_calibration_error as calculate_ece
from src.xgb_baseline.metrics import binary_metrics


ROOT = Path(__file__).resolve().parent
MODELS = ("xgboost", "logistic_regression", "histgradientboosting", "extratrees")
DISPLAY = {
    "xgboost": "XGBoost",
    "logistic_regression": "Logistic Regression",
    "histgradientboosting": "HistGradientBoosting",
    "extratrees": "Extra Trees",
}
FAMILIES = {
    "xgboost": "xgboost",
    "logistic_regression": "logistic_regression",
    "histgradientboosting": "histgb",
    "extratrees": "extratrees",
}
COLORS = {
    "xgboost": "#176B87",
    "logistic_regression": "#B45309",
    "histgradientboosting": "#5B8E7D",
    "extratrees": "#A23B72",
}
SEED = 42
N_BOOT = 2000


def _prediction(model: str, target: str) -> pd.DataFrame:
    path = ROOT / "experiments" / FAMILIES[model] / target / "predictions/test_predictions.csv"
    return pd.read_csv(path)


def _threshold(model: str, target: str) -> float:
    path = ROOT / "experiments" / FAMILIES[model] / target / "metrics/comprehensive_metrics.json"
    return float(json.loads(path.read_text())["selected_threshold"])


def _save_figure(fig, name: str) -> None:
    out = ROOT / "diagrams"
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(out / f"{name}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _metrics_summary() -> pd.DataFrame:
    rows = []
    for target in ("ge_c", "ge_m"):
        for model in MODELS:
            payload = json.loads((ROOT / "experiments" / FAMILIES[model] / target / "metrics/comprehensive_metrics.json").read_text())
            metric = payload["test"]
            intervals = payload["confidence_intervals"]
            rows.append({
                "model": DISPLAY[model],
                "model_key": model,
                "target": target,
                "threshold": payload["selected_threshold"],
                "task": "binary",
                "total_test_samples": metric["total_test_samples"],
                "positive_count": metric["positive_count"],
                "negative_count": metric["negative_count"],
                "confusion_matrix": json.dumps(metric["confusion_matrix"]),
                "accuracy": metric["accuracy"],
                "positive_prevalence": metric["positive_prevalence"],
                **{key: metric[key] for key in ("roc_auc", "pr_auc", "tss", "hss", "csi", "mcc", "f1", "precision", "pod_recall_tpr", "specificity", "far", "brier_score")},
                **{f"{key}_ci_low": intervals[key][0] for key in intervals},
                **{f"{key}_ci_high": intervals[key][1] for key in intervals},
            })
    result = pd.DataFrame(rows)
    result.to_csv(ROOT / "results/results_final.csv", index=False)
    return result


def _paired_auc_comparisons() -> None:
    rows = []
    for target in ("ge_c", "ge_m"):
        frames = {model: _prediction(model, target).sort_values("row_id").reset_index(drop=True) for model in MODELS}
        reference = frames["xgboost"]
        for model, frame in frames.items():
            if not reference.row_id.equals(frame.row_id) or not reference.target.equals(frame.target):
                raise ValueError(f"{model}/{target} does not share the same test rows as XGBoost")
            if not reference.active_region.astype(str).equals(frame.active_region.astype(str)):
                raise ValueError(f"{model}/{target} does not share active-region IDs")
        _, codes, draws = make_cluster_bootstrap_counts(reference.active_region.astype(str), N_BOOT, SEED)
        auc_samples = {}
        auc_point = {}
        for model, frame in frames.items():
            auc_point[model] = float(roc_auc_score(reference.target, frame.probability))
            auc_samples[model] = cluster_bootstrap_metric_samples(
                reference.target.to_numpy(), frame.probability.to_numpy(), codes, draws,
                _threshold(model, target),
            )["roc_auc"]
        for comparator in ("logistic_regression", "histgradientboosting", "extratrees"):
            difference = auc_samples["xgboost"] - auc_samples[comparator]
            finite = difference[np.isfinite(difference)]
            observed = auc_point["xgboost"] - auc_point[comparator]
            low, high = np.percentile(finite, [2.5, 97.5])
            rows.append({
                "model_a": DISPLAY["xgboost"],
                "model_b": DISPLAY[comparator],
                "target": target,
                "auc_a": auc_point["xgboost"],
                "auc_b": auc_point[comparator],
                "difference_a_minus_b": observed,
                "ci_95_low": float(low),
                "ci_95_high": float(high),
                "paired_cluster_bootstrap_p_two_sided": None,
                "bootstrap_resamples": N_BOOT,
                "seed": SEED,
                "unit": "active-region cluster bootstrap; observation-level AUC",
            })
    out = ROOT / "results/statistical_comparisons/auc_comparisons.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)


def _roc_pr_figures() -> None:
    for curve_type in ("roc", "pr"):
        fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.1), sharey=True)
        for panel, target in enumerate(("ge_c", "ge_m")):
            ax = axes[panel]
            for model in MODELS:
                frame = _prediction(model, target)
                y = frame.target.to_numpy()
                p = frame.probability.to_numpy()
                threshold = _threshold(model, target)
                pred = p >= threshold
                if curve_type == "roc":
                    x, value = binary_curve_data(y, p, "roc")
                    auc = roc_auc_score(y, p)
                    ax.plot(x, value, color=COLORS[model], lw=1.35, label=f"{DISPLAY[model]} ({auc:.3f})")
                    op = binary_metrics(y, p, threshold)
                    ax.scatter(1 - op["specificity"], op["pod_recall_tpr"], color=COLORS[model], s=16, zorder=3)
                else:
                    recall, precision = binary_curve_data(y, p, "pr")
                    ap = average_precision_score(y, p)
                    ax.plot(recall, precision, color=COLORS[model], lw=1.35, label=f"{DISPLAY[model]} ({ap:.3f})")
                    op = binary_metrics(y, p, threshold)
                    ax.scatter(op["pod_recall_tpr"], op["precision"], color=COLORS[model], s=16, zorder=3)
            if curve_type == "roc":
                ax.plot([0, 1], [0, 1], color="#555555", ls="--", lw=0.8)
                ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title=f"({chr(97 + panel)}) {target}")
            else:
                prevalence = float(_prediction("xgboost", target).target.mean())
                ax.axhline(prevalence, color="#555555", ls="--", lw=0.8, label=f"Prevalence ({prevalence:.3f})")
                ax.set(xlabel="Recall", ylabel="Precision", title=f"({chr(97 + panel)}) {target}")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=5.8, frameon=False, loc="best")
        fig.tight_layout()
        _save_figure(fig, "figure4_roc_curves" if curve_type == "roc" else "figure5_pr_curves")


def _calibration() -> None:
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.1), sharex=True, sharey=True)
    edges = np.linspace(0, 1, 11)
    for panel, target in enumerate(("ge_c", "ge_m")):
        ax = axes[panel]
        ax.plot([0, 1], [0, 1], color="#555555", ls="--", lw=0.8, label="Ideal")
        for model in MODELS:
            frame = _prediction(model, target)
            y = frame.target.to_numpy()
            p = frame.probability.to_numpy()
            indices = np.minimum(np.digitize(p, edges[1:-1]), 9)
            mean_p, observed, count = [], [], []
            for index in range(10):
                selected = indices == index
                count.append(int(selected.sum()))
                mean_p.append(float(p[selected].mean()) if selected.any() else np.nan)
                observed.append(float(y[selected].mean()) if selected.any() else np.nan)
            ece = calculate_ece(y, p, 10)
            brier = float(np.mean((y - p) ** 2))
            rows.append({"model": DISPLAY[model], "target": target, "ece_equal_width_10": ece, "brier_score": brier, "bin_method": "10 fixed-width bins on [0,1]", "bin_counts": json.dumps(count)})
            ax.plot(mean_p, observed, marker="o", ms=3, lw=1.1, color=COLORS[model], label=f"{DISPLAY[model]} (ECE={ece:.3f})")
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Observed frequency", title=f"({chr(97 + panel)}) {target}")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=5.8, frameon=False, loc="best")
    out = ROOT / "results/calibration"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "calibration_metrics.csv", index=False)
    fig.tight_layout()
    _save_figure(fig, "figure7_calibration")


def _threshold_sensitivity() -> None:
    out = ROOT / "results/threshold_sensitivity"
    out.mkdir(parents=True, exist_ok=True)
    thresholds = (0.05, 0.10, 0.20, 0.30, 0.50)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.1), sharey=True)
    for panel, target in enumerate(("ge_m", "ge_c")):
        frame = _prediction("xgboost", target)
        records = []
        for threshold in thresholds:
            metrics = binary_metrics(frame.target, frame.probability, threshold)
            records.append({"threshold": threshold, "precision": metrics["precision"], "recall": metrics["pod_recall_tpr"], "far": metrics["far"], "tss": metrics["tss"], "f1": metrics["f1"], "frozen_validation_threshold": _threshold("xgboost", target)})
        pd.DataFrame(records).to_csv(out / f"xgboost_{target}.csv", index=False)
        chart = pd.DataFrame(records)
        ax = axes[panel]
        for key in ("precision", "recall", "far", "tss", "f1"):
            ax.plot(chart.threshold, chart[key], marker="o", ms=3, lw=1, label=key.upper() if key == "far" else key.title())
        ax.axvline(_threshold("xgboost", target), color="#555555", ls="--", lw=0.8)
        ax.set(xlabel="Descriptive test threshold", ylabel="Metric value", ylim=(0, 1), title=f"({chr(97 + panel)}) {target}")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=5.8, frameon=False, ncol=2)
    fig.tight_layout()
    _save_figure(fig, "figure6_threshold_sensitivity")


def _dotwhiskers(summary: pd.DataFrame) -> None:
    metric_specs = (("roc_auc", "ROC-AUC", "figure9_roc_auc_dotwhisker"),
                    ("pr_auc", "PR-AUC", "figure10_pr_auc_dotwhisker"),
                    ("tss", "TSS", "figure11_tss_dotwhisker"))
    for metric, title, filename in metric_specs:
        fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.1), sharex=True)
        for panel, target in enumerate(("ge_c", "ge_m")):
            ax = axes[panel]
            rows = summary[summary.target == target].set_index("model_key").reindex(MODELS)
            for position, model in enumerate(MODELS):
                row = rows.loc[model]
                low = row[f"{metric}_ci_low"]
                high = row[f"{metric}_ci_high"]
                ax.errorbar(row[metric], position, xerr=[[row[metric] - low], [high - row[metric]]], fmt="o", color=COLORS[model], capsize=2, ms=4)
            ax.set_yticks(range(len(MODELS)), [DISPLAY[model] for model in MODELS])
            ax.set(xlim=(0, 1), xlabel=title, title=f"({chr(97 + panel)}) {target}")
            ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        _save_figure(fig, filename)


def _baseline_comparison() -> None:
    rows = []
    for target in ("ge_c", "ge_m"):
        for model in MODELS:
            frame = _prediction(model, target)
            metrics = json.loads((ROOT / "experiments" / FAMILIES[model] / target / "metrics/comprehensive_metrics.json").read_text())["test"]
            rows.append({"model": DISPLAY[model], "target": target, "baseline_type": "trained_model", "roc_auc": metrics["roc_auc"], "pr_auc": metrics["pr_auc"], "tss": metrics["tss"], "f1": metrics["f1"], "positive_prevalence": float(frame.target.mean())})
        frame = _prediction("xgboost", target)
        majority = pd.read_csv(ROOT / "experiments/xgboost" / target / "predictions/test_majority_climatology.csv")
        prevalence = float(frame.target.mean())
        rows.append({"model": "Majority climatology", "target": target, "baseline_type": "training-prevalence constant probability", "roc_auc": 0.5, "pr_auc": average_precision_score(frame.target, majority.probability), "tss": np.nan, "f1": np.nan, "positive_prevalence": prevalence})
        rows.append({"model": "No-skill PR reference", "target": target, "baseline_type": "test-set positive prevalence reference, not a fitted predictor", "roc_auc": np.nan, "pr_auc": prevalence, "tss": np.nan, "f1": np.nan, "positive_prevalence": prevalence})
    pd.DataFrame(rows).to_csv(ROOT / "results/baseline_comparison.csv", index=False)


def _cnn_lstm_artifacts() -> None:
    source = json.loads((ROOT / "cnn_lstm_test_results.json").read_text())
    prediction_path = ROOT / "experiments/cnn_lstm/multiclass/predictions/test_predictions.csv"
    prediction_frame = pd.read_csv(prediction_path) if prediction_path.exists() else None
    if prediction_frame is not None:
        label_ids = {name: index for index, name in enumerate(("C", "M", "X"))}
        labels = prediction_frame.target.map(label_ids).to_numpy(dtype=np.int64)
        predictions = prediction_frame.predicted_label.map(label_ids).to_numpy(dtype=np.int64)
        matrix = confusion_matrix(labels, predictions, labels=[0, 1, 2])
    else:
        matrix = np.asarray(source["confusion_matrix"], dtype=np.int64)
        labels = np.repeat(np.arange(3), matrix.sum(axis=1))
        predictions = np.concatenate([
            np.repeat(np.arange(3), matrix[row]) for row in range(3)
        ])
    if matrix.shape != (3, 3):
        raise ValueError("Expected saved 3-class CNN-LSTM confusion matrix")
    report = classification_report(labels, predictions, labels=[0, 1, 2], output_dict=True, zero_division=0)
    per_class = {}
    totals = int(matrix.sum())
    for index, name in enumerate(("C", "M", "X")):
        tp = int(matrix[index, index])
        fn = int(matrix[index].sum() - tp)
        fp = int(matrix[:, index].sum() - tp)
        tn = totals - tp - fn - fp
        recall = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        far = fp / (tp + fp) if tp + fp else 0.0
        csi = tp / (tp + fn + fp) if tp + fn + fp else 0.0
        hss_denominator = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
        hss = 2 * (tp * tn - fn * fp) / hss_denominator if hss_denominator else 0.0
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": float(report[str(index)]["f1-score"]),
            "support": int(matrix[index].sum()),
            "tss": recall + specificity - 1,
            "hss": hss,
            "csi": csi,
            "far": far,
            "specificity": specificity,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        }
    metrics = {
        "model": "CNN-LSTM",
        "task": "multiclass C/M/X",
        "test_sequences": totals,
        "class_order": ["C", "M", "X"],
        "class_counts": {name: int(matrix[index].sum()) for index, name in enumerate(("C", "M", "X"))},
        "confusion_matrix": matrix.tolist(),
        "row_normalized_confusion_matrix": (matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)).tolist(),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "multiclass_mcc": float(matthews_corrcoef(labels, predictions)),
        "macro_tss": float(np.mean([value["tss"] for value in per_class.values()])),
        "macro_hss": float(np.mean([value["hss"] for value in per_class.values()])),
        "macro_csi": float(np.mean([value["csi"] for value in per_class.values()])),
        "macro_far": float(np.mean([value["far"] for value in per_class.values()])),
        "per_class": per_class,
        "error_flows": {"C_to_M": int(matrix[0, 1]), "M_to_C": int(matrix[1, 0]), "X_to_C": int(matrix[2, 0]), "X_to_M": int(matrix[2, 1])},
        "probability_metrics": None,
        "uncertainty": None,
        "limitations": None,
    }
    if prediction_frame is not None:
        probability_columns = [f"probability_{name}" for name in ("C", "M", "X")]
        probability = prediction_frame[probability_columns].to_numpy(dtype=np.float64)
        class_probabilities = {}
        bootstrap_intervals = {}
        _, row_codes, region_draws = make_cluster_bootstrap_counts(
            prediction_frame.ar.astype(str).to_numpy(), n_resamples=2000, seed=42
        )
        hard_samples = multiclass_cluster_bootstrap_metric_samples(
            labels, predictions, row_codes, region_draws
        )
        bootstrap_intervals.update(percentile_intervals(hard_samples))
        auc_values = []
        ap_values = []
        auc_bootstrap_by_class = []
        ap_bootstrap_by_class = []
        for index, name in enumerate(("C", "M", "X")):
            binary_target = (labels == index).astype(np.int8)
            auc = float(roc_auc_score(binary_target, probability[:, index]))
            average_precision = float(average_precision_score(binary_target, probability[:, index]))
            class_probabilities[name] = {"roc_auc_ovr": auc, "pr_auc_average_precision_ovr": average_precision}
            auc_values.append(auc)
            ap_values.append(average_precision)
            samples = cluster_bootstrap_metric_samples(
                binary_target,
                probability[:, index],
                row_codes,
                region_draws,
                0.5,
            )
            auc_bootstrap_by_class.append(samples["roc_auc"])
            ap_bootstrap_by_class.append(samples["pr_auc"])
            bootstrap_intervals[f"{name}_roc_auc_ovr"] = percentile_intervals({"roc_auc": samples["roc_auc"]})["roc_auc"]
            bootstrap_intervals[f"{name}_pr_auc_ovr"] = percentile_intervals({"pr_auc": samples["pr_auc"]})["pr_auc"]
        macro_auc_samples = np.nanmean(np.vstack(auc_bootstrap_by_class), axis=0)
        macro_ap_samples = np.nanmean(np.vstack(ap_bootstrap_by_class), axis=0)
        macro_auc_interval = percentile_intervals({"roc_auc": macro_auc_samples})["roc_auc"]
        macro_ap_interval = percentile_intervals({"pr_auc": macro_ap_samples})["pr_auc"]
        metrics["probability_metrics"] = {
            "classwise_ovr": class_probabilities,
            "macro_roc_auc_ovr": float(np.mean(auc_values)),
            "macro_pr_auc_average_precision_ovr": float(np.mean(ap_values)),
            "macro_roc_auc_ovr_ci_95": macro_auc_interval,
            "macro_pr_auc_average_precision_ovr_ci_95": macro_ap_interval,
        }
        metrics["uncertainty"] = {
            "resamples": 2000,
            "seed": 42,
            "method": "active-region cluster bootstrap; percentile 95% intervals",
            "cluster_count": int(prediction_frame.ar.nunique()),
            "ci_95": bootstrap_intervals,
        }
        metrics["gpu_inference"] = {
            "device": json.loads((ROOT / "experiments/cnn_lstm/multiclass/metrics/gpu_test_metrics.json").read_text())["device"],
            "probabilities_csv": str(prediction_path.relative_to(ROOT)),
        }
    else:
        metrics["limitations"] = "Only aggregate confusion counts are saved; no per-sequence probabilities or predictions are available for ROC/PR, calibration, or cluster bootstrap."
    (ROOT / "results/cnn_lstm_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    for normalized, name in ((False, "figure3_confusion_matrix"), (True, "figure3_confusion_matrix_normalized")):
        values = metrics["row_normalized_confusion_matrix"] if normalized else matrix
        fig, ax = plt.subplots(figsize=(3.5, 3.1))
        image = ax.imshow(values, cmap="YlGnBu", vmin=0, vmax=1 if normalized else None)
        for row in range(3):
            for column in range(3):
                text = f"{values[row][column]:.3f}" if normalized else str(values[row][column])
                ax.text(column, row, text, ha="center", va="center", color="#111111", fontsize=8)
        ax.set(xticks=range(3), yticks=range(3), xticklabels=("C", "M", "X"), yticklabels=("C", "M", "X"), xlabel="Predicted class", ylabel="True class", title="Row-normalized" if normalized else "Test confusion matrix")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        _save_figure(fig, name)

    sequence_labels = pd.read_csv(ROOT / "data/Test_sequences.csv", usecols=["target"]).target
    image_counts = sequence_labels.value_counts().reindex(("C", "M", "X"), fill_value=0)
    rows = [{"task": "CNN-LSTM", "class": name, "count": int(image_counts[name]), "prevalence": float(image_counts[name] / len(sequence_labels))} for name in ("C", "M", "X")]
    binary = {}
    for target in ("ge_c", "ge_m"):
        frame = _prediction("xgboost", target)
        counts = frame.target.value_counts().reindex((0, 1), fill_value=0)
        binary[target] = counts
        rows.extend([{"task": target, "class": "negative", "count": int(counts[0]), "prevalence": float(counts[0] / len(frame))}, {"task": target, "class": "positive", "count": int(counts[1]), "prevalence": float(counts[1] / len(frame))}])
    pd.DataFrame(rows).to_csv(ROOT / "results/class_distribution.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.8))
    palettes = (("#176B87", "#B45309", "#A23B72"), ("#648C7A", "#B45309"), ("#648C7A", "#A23B72"))
    axes[0].bar(("C", "M", "X"), image_counts.to_numpy(), color=palettes[0])
    axes[0].set(title="CNN-LSTM test classes", ylabel="Sequences")
    for axis, target, palette in zip(axes[1:], ("ge_c", "ge_m"), palettes[1:]):
        counts = binary[target]
        axis.bar(("Negative", "Positive"), [counts[0], counts[1]], color=palette)
        axis.set(title=f"{target} test labels")
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    _save_figure(fig, "figure2_class_distribution")

    results_path = ROOT / "results/results_final.csv"
    existing = pd.read_csv(results_path)
    existing = existing.loc[~existing.model.eq("CNN-LSTM")]
    cnn_row = {
        "model": "CNN-LSTM",
        "model_key": "cnn_lstm",
        "target": "multiclass_C_M_X",
        "threshold": np.nan,
        "task": "multiclass",
        "total_test_samples": totals,
        "positive_count": np.nan,
        "negative_count": np.nan,
        "confusion_matrix": json.dumps(matrix.tolist()),
        "accuracy": metrics["accuracy"],
        "positive_prevalence": np.nan,
        "roc_auc": np.nan,
        "pr_auc": np.nan,
        "tss": metrics["macro_tss"],
        "hss": metrics["macro_hss"],
        "csi": metrics["macro_csi"],
        "mcc": metrics["multiclass_mcc"],
        "f1": metrics["macro_f1"],
        "precision": metrics["macro_precision"],
        "pod_recall_tpr": metrics["macro_recall"],
        "specificity": np.nan,
        "far": metrics["macro_far"],
        "brier_score": np.nan,
        "weighted_f1": metrics["weighted_f1"],
        "macro_roc_auc_ovr": metrics["probability_metrics"]["macro_roc_auc_ovr"] if metrics["probability_metrics"] else np.nan,
        "macro_pr_auc_average_precision_ovr": metrics["probability_metrics"]["macro_pr_auc_average_precision_ovr"] if metrics["probability_metrics"] else np.nan,
    }
    if metrics["uncertainty"]:
        for metric, interval in metrics["uncertainty"]["ci_95"].items():
            cnn_row[f"{metric}_ci_low"], cnn_row[f"{metric}_ci_high"] = interval
        probability_metrics = metrics["probability_metrics"]
        cnn_row["macro_roc_auc_ovr_ci_low"], cnn_row["macro_roc_auc_ovr_ci_high"] = probability_metrics["macro_roc_auc_ovr_ci_95"]
        cnn_row["macro_pr_auc_average_precision_ovr_ci_low"], cnn_row["macro_pr_auc_average_precision_ovr_ci_high"] = probability_metrics["macro_pr_auc_average_precision_ovr_ci_95"]
    existing = pd.concat([existing, pd.DataFrame([cnn_row])], ignore_index=True)
    existing.to_csv(results_path, index=False)


def _workflow_figure() -> None:
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    old_svg_fonttype = plt.rcParams["svg.fonttype"]
    old_pdf_fonttype = plt.rcParams["pdf.fonttype"]
    old_font_family = plt.rcParams["font.family"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.family"] = "Liberation Sans"
    fig, ax = plt.subplots(figsize=(7.1, 5.15), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 7.1)
    ax.set_ylim(-0.42, 6.15)
    ax.set_axis_off()

    ink = "#263746"
    muted = "#536573"
    image_fill = "#EDF3F6"
    table_fill = "#F2F3EF"
    shared_fill = "#F5F6F7"
    metric_fill = "#FAFBFC"
    line_width = 0.75

    def box(x, y, width, height, title, detail=(), fill=shared_fill, title_size=8.0, detail_size=6.5, title_weight="semibold"):
        patch = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.025,rounding_size=0.045",
            linewidth=line_width,
            edgecolor=ink,
            facecolor=fill,
            zorder=2,
        )
        ax.add_patch(patch)
        detail = tuple(detail)
        title_y = y + height * (0.73 if detail else 0.50)
        ax.text(x + width / 2, title_y, title, ha="center", va="center", color=ink,
                fontsize=title_size, fontweight=title_weight, linespacing=1.08, zorder=3)
        if detail:
            detail_y = y + height * (0.25 if len(detail) > 1 else 0.27)
            ax.text(x + width / 2, detail_y, "\n".join(detail), ha="center", va="center",
                color=muted, fontsize=detail_size, linespacing=1.12, zorder=3)

    def arrow(start, end, *, color=ink, width=0.85, scale=8, zorder=1):
        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=scale,
            linewidth=width, color=color, shrinkA=0, shrinkB=0,
            connectionstyle="arc3,rad=0", zorder=zorder,
        ))

    ax.text(3.55, 6.015, "Experimental Workflow for Solar Flare Prediction",
            ha="center", va="center", fontsize=11.0, fontweight="bold", color=ink)

    box(1.30, 5.48, 4.50, 0.40, "HMI active-region observations",
        ("950,047 observations  |  1,570 active regions  |  29 engineered numeric features",),
        fill=shared_fill, title_size=8.4, detail_size=6.4)
    arrow((3.55, 5.46), (3.55, 5.20))
    box(1.86, 4.79, 3.38, 0.40, "Active-region grouped split",
        ("Train / Validation / Test: 759,357 / 95,933 / 94,757 observations",),
        fill=shared_fill, title_size=8.2, detail_size=6.3)

    # Explicit fan-out from one split node: both representations are peers.
    ax.plot([3.55, 3.55], [4.77, 4.55], color=ink, linewidth=0.85, zorder=1)
    ax.plot([1.82, 5.28], [4.55, 4.55], color=ink, linewidth=0.85, zorder=1)
    arrow((1.82, 4.55), (1.82, 4.08))
    arrow((5.28, 4.55), (5.28, 4.08))

    box(0.28, 3.64, 3.08, 0.42, "IMAGE-SEQUENCE BRANCH",
        ("10-frame chronological image sequences",), fill=image_fill, title_size=7.8, detail_size=6.2)
    box(3.74, 3.64, 3.08, 0.42, "TABULAR BRANCH",
        ("29 engineered numeric features; separate table",), fill=table_fill, title_size=7.8, detail_size=6.2)

    box(0.28, 2.47, 3.08, 0.91, "CNN-LSTM  |  multiclass C / M / X",
        ("Current training: focal loss (gamma 0.8)",
         "inverse-square-root class sampling + class weights",
         "Checkpoint: cnn_lstm_best.pth",
         "GPU inference: PyTorch 2.11.0 + ROCm 7.2"),
        fill=image_fill, title_size=7.8, detail_size=6.0)
    box(3.74, 2.47, 3.08, 0.91, "Tabular classifiers",
        ("XGBoost  |  Logistic Regression",
         "HistGradientBoosting  |  Extra Trees",
         "ge_c: C/M/X vs 0  |  ge_m: M/X vs 0/C"),
        fill=table_fill, title_size=8.0, detail_size=6.2)
    arrow((1.82, 3.62), (1.82, 3.41))
    arrow((5.28, 3.62), (5.28, 3.41))

    box(0.63, 1.89, 2.38, 0.34, "Class probabilities",
        ("2,021 test sequences  |  C / M / X",), fill=image_fill, title_size=7.5, detail_size=5.9)
    box(4.09, 1.89, 2.38, 0.34, "Event probabilities",
        ("Validation threshold frozen for Test",), fill=table_fill, title_size=7.5, detail_size=5.8)
    arrow((1.82, 2.45), (1.82, 2.25))
    arrow((5.28, 2.45), (5.28, 2.25))

    # The two probability outputs converge only at common held-out evaluation.
    ax.plot([1.82, 1.82], [1.87, 1.69], color=ink, linewidth=0.85, zorder=1)
    ax.plot([5.28, 5.28], [1.87, 1.69], color=ink, linewidth=0.85, zorder=1)
    ax.plot([1.82, 5.28], [1.69, 1.69], color=ink, linewidth=0.85, zorder=1)
    arrow((3.55, 1.69), (3.55, 1.49))
    box(2.13, 1.10, 2.84, 0.38, "Held-out Test evaluation",
        ("Separate task-specific predictions; no matched comparison",),
        fill=shared_fill, title_size=8.1, detail_size=6.0)

    ax.plot([3.55, 3.55], [1.08, 0.97], color=ink, linewidth=0.85, zorder=1)
    ax.plot([1.31, 5.79], [0.97, 0.97], color=ink, linewidth=0.85, zorder=1)
    metric_centers = (1.31, 3.55, 5.79)
    for center in metric_centers:
        arrow((center, 0.97), (center, 0.84))
    box(0.28, 0.42, 2.06, 0.41, "ROC / PR analysis",
        ("From task-specific probabilities",), fill=metric_fill, title_size=7.4, detail_size=5.8)
    box(2.52, 0.42, 2.06, 0.41, "Calibration",
        ("Tabular Brier score / ECE",), fill=metric_fill, title_size=7.4, detail_size=5.8)
    box(4.76, 0.42, 2.06, 0.41, "Threshold metrics",
        ("Tabular POD / FAR / CSI / F1", "TSS / HSS / MCC"), fill=metric_fill, title_size=7.2, detail_size=5.1)

    ax.plot([1.31, 1.31], [0.40, 0.30], color=ink, linewidth=0.75, zorder=1)
    ax.plot([3.55, 3.55], [0.40, 0.30], color=ink, linewidth=0.75, zorder=1)
    ax.plot([5.79, 5.79], [0.40, 0.30], color=ink, linewidth=0.75, zorder=1)
    ax.plot([1.31, 5.79], [0.30, 0.30], color=ink, linewidth=0.75, zorder=1)
    arrow((3.55, 0.30), (3.55, 0.21), scale=7)
    box(1.85, 0.02, 3.40, 0.18, "2,000-replicate active-region cluster bootstrap  |  95% CIs",
        fill=shared_fill, title_size=6.8)

    ax.text(3.55, -0.12,
            "Tabular thresholds: Validation maximum TSS over 0.1-0.9; frozen for Test.",
            ha="center", va="center", fontsize=6.0, color=muted, clip_on=False)
    ax.text(3.55, -0.25,
            "Image and engineered-feature pipelines are complementary, non-matched evaluation tasks.",
            ha="center", va="center", fontsize=6.0, color=muted, clip_on=False)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.995, bottom=0.055)
    _save_figure(fig, "figure1_workflow")
    plt.rcParams["svg.fonttype"] = old_svg_fonttype
    plt.rcParams["pdf.fonttype"] = old_pdf_fonttype
    plt.rcParams["font.family"] = old_font_family


def _literature_table() -> None:
    rows = [
        {"study": "Nishizuka et al.", "dataset": "--", "forecast_horizon": "--", "target": "--", "representation": "--", "split_strategy": "--", "model": "--", "tss": "--", "hss": "--", "pr_auc": "--", "notes": "Cited in the compiled report; paper-specific values were not extracted and verified."},
        {"study": "Florios et al.", "dataset": "--", "forecast_horizon": "--", "target": "--", "representation": "--", "split_strategy": "--", "model": "--", "tss": "--", "hss": "--", "pr_auc": "--", "notes": "Cited in the compiled report; paper-specific values were not extracted and verified."},
        {"study": "Sun et al.", "dataset": "--", "forecast_horizon": "--", "target": "--", "representation": "--", "split_strategy": "--", "model": "--", "tss": "--", "hss": "--", "pr_auc": "--", "notes": "Cited in the compiled report; paper-specific values were not extracted and verified."},
        {"study": "Present study", "dataset": "Dryad-derived reduced-resolution HMI", "forecast_horizon": "--; exact label horizon not verified from artifacts", "target": "ge_c, ge_m; separate C/M/X sequence task", "representation": "29 generic engineered columns; 10-frame image sequences", "split_strategy": "Predefined active-region grouped; timestamp ranges overlap", "model": "XGBoost, Logistic Regression, HistGradientBoosting, Extra Trees, CNN-LSTM", "tss": "See results_final.csv", "hss": "See results_final.csv", "pr_auc": "See results_final.csv", "notes": "Direct ranking is not justified when labels, horizons, data periods, splits, thresholds, or features differ."},
    ]
    out = ROOT / "results/literature_comparison.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)


def _feature_importance() -> None:
    out = ROOT / "results/feature_importance"
    out.mkdir(parents=True, exist_ok=True)
    for target in ("ge_c", "ge_m"):
        model_path = ROOT / "experiments/xgboost" / target / "models/xgboost.json"
        model = json.loads(model_path.read_text())
        learner = model["learner"]
        feature_names = learner.get("feature_names", [])
        trees = learner["gradient_booster"]["model"]["trees"]
        if len(feature_names) != 29 or not trees:
            raise ValueError(f"Unexpected saved XGBoost model structure for {target}")
        gains = np.zeros(len(feature_names), dtype=np.float64)
        frequencies = np.zeros(len(feature_names), dtype=np.int64)
        for tree in trees:
            splits = np.asarray(tree["split_indices"], dtype=np.int64)
            left = np.asarray(tree["left_children"], dtype=np.int64)
            right = np.asarray(tree["right_children"], dtype=np.int64)
            loss_changes = np.asarray(tree["loss_changes"], dtype=np.float64)
            internal = (left >= 0) & (right >= 0)
            for index, gain in zip(splits[internal], loss_changes[internal]):
                if index < 0 or index >= len(feature_names):
                    raise ValueError(f"Invalid feature index {index} in {model_path}")
                gains[index] += gain
                frequencies[index] += 1
        mean_gain = np.divide(gains, frequencies, out=np.zeros_like(gains), where=frequencies > 0)
        table = pd.DataFrame({"feature_index": np.arange(len(feature_names)), "feature": feature_names, "mean_gain": mean_gain, "total_gain": gains, "split_frequency": frequencies})
        table = table.sort_values("total_gain", ascending=False)
        table.to_csv(out / f"xgboost_{target}_gain.csv", index=False)
        selected = table.head(20).sort_values("total_gain")
        fig, ax = plt.subplots(figsize=(5.6, 4.0))
        ax.barh(selected.feature, selected.total_gain, color="#176B87")
        ax.set(xlabel="Total split gain (saved XGBoost tree statistics)", ylabel="Generic feature column", title=f"XGBoost feature importance: {target}")
        ax.tick_params(axis="y", labelsize=6.5)
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        _save_figure(fig, f"figure8_feature_importance_{target}")


def _write_final_report(summary: pd.DataFrame) -> None:
    cnn = json.loads((ROOT / "results/cnn_lstm_metrics.json").read_text())
    cnn_auc = cnn["probability_metrics"]
    cnn_uncertainty = cnn["uncertainty"]
    gpu_run = json.loads((ROOT / "experiments/cnn_lstm/multiclass/metrics/gpu_test_metrics.json").read_text())
    calibration = pd.read_csv(ROOT / "results/calibration/calibration_metrics.csv")
    comparisons = pd.read_csv(ROOT / "results/statistical_comparisons/auc_comparisons.csv")
    threshold_m = pd.read_csv(ROOT / "results/threshold_sensitivity/xgboost_ge_m.csv")
    baseline = pd.read_csv(ROOT / "results/baseline_comparison.csv")
    lines = [
        "# Final Results: Solar-Flare Benchmark",
        "",
        "This report summarizes verified outputs produced from the repository's saved artifacts. It separates observation-level point estimates from active-region-cluster bootstrap uncertainty. The NASA/SHARP CSV was not used for training or evaluation.",
        "",
        "## Dataset and protocol",
        "",
        "The feature file and flare-label file each contain 950,047 rows and 29 generic numeric feature columns. The predefined grouped manifests contain 759,357 Train, 95,933 Validation, and 94,757 Test rows. Test predictions cover 157 active regions. The split loader checks for active-region overlap, but timestamp ranges overlap, so the evaluation is grouped benchmark performance, not chronological forecasting.",
        "",
        "For each tabular model/target, the threshold is selected on Validation from 0.1 to 0.9 by maximum TSS, with recall as the tie-break, then frozen for Test. Model comparisons use the same 94,757 test rows and identical labels/active-region IDs. Point estimates below are observation-level. The 95% intervals resample 157 active regions with replacement, seed 42, for 2,000 replicates; all rows from a drawn region travel together.",
        "",
        "## CNN-LSTM multiclass results",
        "",
        f"The 2,021-sequence test distribution is C={cnn['class_counts']['C']}, M={cnn['class_counts']['M']}, X={cnn['class_counts']['X']}. Macro-F1 is {cnn['macro_f1']:.3f}, macro recall {cnn['macro_recall']:.3f}, macro precision {cnn['macro_precision']:.3f}, weighted F1 {cnn['weighted_f1']:.3f}, macro TSS {cnn['macro_tss']:.3f}, macro HSS {cnn['macro_hss']:.3f}, macro CSI {cnn['macro_csi']:.3f}, macro FAR {cnn['macro_far']:.3f}, and multiclass MCC {cnn['multiclass_mcc']:.3f}. Accuracy is {cnn['accuracy']:.3f} and is secondary because C dominates the sample.",
        "",
        "| True class | Precision | Recall | F1 | ROC-AUC OVR | PR-AUC OVR | Support |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("C", "M", "X"):
        item = cnn["per_class"][name]
        auc_item = cnn_auc["classwise_ovr"][name]
        lines.append(f"| {name} | {item['precision']:.3f} | {item['recall']:.3f} | {item['f1']:.3f} | {auc_item['roc_auc_ovr']:.3f} | {auc_item['pr_auc_average_precision_ovr']:.3f} | {item['support']} |")
    flows = cnn["error_flows"]
    lines.extend([
        "",
        f"The held-out confusion matrix records C-to-M={flows['C_to_M']}, M-to-C={flows['M_to_C']}, X-to-C={flows['X_to_C']}, and X-to-M={flows['X_to_M']}; it identified {cnn['per_class']['X']['tp']} of the 34 X examples. Macro one-vs-rest ROC-AUC is {cnn_auc['macro_roc_auc_ovr']:.3f} and macro average precision is {cnn_auc['macro_pr_auc_average_precision_ovr']:.3f}. Active-region cluster-bootstrap 95% intervals for those macro values are [{cnn_auc['macro_roc_auc_ovr_ci_95'][0]:.3f}, {cnn_auc['macro_roc_auc_ovr_ci_95'][1]:.3f}] and [{cnn_auc['macro_pr_auc_average_precision_ovr_ci_95'][0]:.3f}, {cnn_auc['macro_pr_auc_average_precision_ovr_ci_95'][1]:.3f}], respectively. All CNN intervals use {cnn_uncertainty['cluster_count']} test active regions and {cnn_uncertainty['resamples']} bootstrap replicates.",
        "",
        f"GPU inference with the existing best checkpoint completed on {gpu_run['device']} using PyTorch {gpu_run['torch_version']} in {gpu_run['elapsed_inference_seconds']:.1f} seconds. CPU/GPU probabilities agreed within 1.8e-7 on the first four identical test samples. The checkpoint configuration already uses focal loss (gamma 0.8), inverse-square-root class sampling, and class weights, so it is not an unweighted baseline. No controlled new imbalance variants were run in this pass.",
        "",
        "## Binary model results",
        "",
        "All values below are observation-level on the same held-out rows. Confidence intervals are active-region cluster-bootstrap percentile intervals. The complete 12-metric JSON per model and target includes counts, prevalence, confusion matrix, and intervals for every metric.",
        "",
        "| Target | Model | Prevalence | Threshold | ROC-AUC (95% CI) | PR-AUC (95% CI) | TSS (95% CI) | HSS | CSI | MCC | F1 | Precision | Recall | Specificity | FAR | Brier |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for _, row in summary.iterrows():
        lines.append(
            f"| {row.target} | {row.model} | {row.positive_prevalence:.3f} | {row.threshold:.1f} "
            f"| {row.roc_auc:.3f} ({row.roc_auc_ci_low:.3f}-{row.roc_auc_ci_high:.3f}) "
            f"| {row.pr_auc:.3f} ({row.pr_auc_ci_low:.3f}-{row.pr_auc_ci_high:.3f}) "
            f"| {row.tss:.3f} ({row.tss_ci_low:.3f}-{row.tss_ci_high:.3f}) "
            f"| {row.hss:.3f} | {row.csi:.3f} | {row.mcc:.3f} | {row.f1:.3f} | {row.precision:.3f} "
            f"| {row.pod_recall_tpr:.3f} | {row.specificity:.3f} | {row.far:.3f} | {row.brier_score:.3f} |"
        )
    lines.extend([
        "",
        "The positive prevalence is 0.227 for ge_c and 0.035 for ge_m. High ranking performance does not imply a favorable operating point: at ge_m's frozen XGBoost threshold 0.1, recall is 0.642 and FAR is 0.725, with precision 0.275. Thresholds were not changed using this sensitivity analysis.",
        "",
        "## Paired AUC comparisons",
        "",
        "Differences are XGBoost minus comparator, paired by active-region bootstrap on identical observation-level test rows. No p-values are reported; the paired 95% intervals are used descriptively and no inferential test claim is made.",
        "",
        "| Target | Comparison | AUC difference | Paired 95% CI |",
        "|---|---|---:|---:|",
    ])
    for _, row in comparisons.iterrows():
        lines.append(f"| {row.target} | {row.model_a} vs {row.model_b} | {row.difference_a_minus_b:.3f} | [{row.ci_95_low:.4f}, {row.ci_95_high:.4f}] |")
    lines.extend([
        "",
        "The paired intervals exclude zero for XGBoost versus Extra Trees on ge_c and XGBoost versus HistGradientBoosting on ge_m. The other four intervals cross zero. No p-values or multiplicity-adjusted significance claims are made.",
        "",
        "## Calibration and baselines",
        "",
        "Calibration uses 10 equal-width bins on [0,1]; ECE is the bin-frequency-weighted absolute difference between mean predicted probability and observed frequency. This is distinct from discrimination (ROC-AUC/PR-AUC). Brier score and ECE are in the table below; no recalibration was performed.",
        "",
        "| Target | Model | Brier | ECE |",
        "|---|---|---:|---:|",
    ])
    for _, row in calibration.iterrows():
        lines.append(f"| {row.target} | {row.model} | {row.brier_score:.3f} | {row.ece_equal_width_10:.3f} |")
    lines.extend([
        "",
        "The majority-climatology baseline assigns a constant training-prevalence score, giving ROC-AUC 0.5 and average precision equal to observed test prevalence; the no-skill PR reference is also test prevalence. These baselines are saved in `results/baseline_comparison.csv`. CNN-LSTM accuracy alone is not an adequate comparison under this class imbalance.",
        "",
        "## Threshold sensitivity",
        "",
        "The ge_m sweep is descriptive on Test; the selected validation threshold remains 0.1. At thresholds 0.05, 0.10, 0.20, 0.30, and 0.50, XGBoost precision is " + ", ".join(f"{value:.3f}" for value in threshold_m.precision) + " and recall is " + ", ".join(f"{value:.3f}" for value in threshold_m.recall) + ". Equivalent ge_c values are saved alongside it.",
        "",
        "## Feature importance and ablation",
        "",
        "Valid gain and split-frequency importance was reconstructed from the saved XGBoost JSON tree statistics because the pre-existing importance CSVs contain only zeros. Each saved model used 27 of 29 columns. Feature names are generic feature_00 through feature_28; the source contains no validated physical-name mapping. Importance is therefore shown only by generic column ID, not interpreted as gradient/PIL/wavelet/flux evidence. Feature-group ablations and SHAP were not performed: group mapping is ambiguous and XGBoost is unavailable in the fresh environment.",
        "",
        "## Feasibility and limitations",
        "",
        "- The primary tabular prediction files are real saved probabilities; HistGradientBoosting and Extra Trees probabilities were generated from existing fitted models, and their thresholds/point metrics were checked against existing records. No tabular model was retrained.",
        "- The saved HistGradientBoosting/Extra Trees joblib models were serialized under scikit-learn 1.9.0 and loaded under 1.9.1, which emitted version warnings. HistGradientBoosting reproduces its stored metrics; regenerated Extra Trees ROC-AUC/AP differ from the historical metric JSON by less than 0.000004. The new saved probabilities and metrics are the values used here.",
        "- No chronological experiment was run. Existing split timestamp ranges overlap; a separate grouped chronological protocol would require a new, predeclared split and new training runs.",
        "- The CNN sequence test has 2,021 C/M/X examples and no no-flare class. Although all 2,021 sequence endpoints occur in the tabular test file, the trained CNN's multiclass task is not the ge_c/ge_m binary task and does not include negatives. A valid matched binary comparison is therefore not supported by the current CNN output.",
        "- The CNN already uses imbalance-aware training. Additional controlled variants were not run in this GPU inference pass; no claim of improvement is made.",
        "- The existing XGBoost headline AUCs 0.891 (ge_c) and 0.939 (ge_m) were active-region-aggregated results, while comparison model values were observation-level. The consistent observation-level XGBoost results are 0.839 and 0.904. The two units must not be mixed in cross-model ranking.",
        "- The NASA/SHARP CSV remains separate and was not used. No external-validation results are claimed.",
        "- There is no editable IEEE LaTeX source in the repository, only `Sun-Is-A-Deadly-Laser.pdf`. The compiled PDF was not altered. A manuscript revision requires the source files.",
        "",
        "## Artifact locations",
        "",
        "Complete predictions and metrics are under `experiments/<model>/<target>/predictions/` and `experiments/<model>/<target>/metrics/comprehensive_metrics.json`; the Phase 0 inventory is `results/PHASE0_ARTIFACT_AUDIT.md`. Statistical intervals/comparisons, calibration, baselines, threshold sweeps, and importance tables are under `results/`. Publication figures are in `diagrams/`, with PNG, PDF, and SVG versions for Figures 1-11 except Figure 8, which is available only for the two XGBoost importance panels. Figure 12 was not generated because no chronological experiment exists.",
        "",
        "## Reproduction",
        "",
        "Run one model-target analysis at a time, then regenerate aggregate outputs:",
        "",
        "```bash",
        "source .research-venv/bin/activate",
        "python build_saved_predictions.py --target ge_c --model histgradientboosting",
        "python build_saved_predictions.py --target ge_m --model histgradientboosting",
        "python build_saved_predictions.py --target ge_c --model extratrees",
        "python build_saved_predictions.py --target ge_m --model extratrees",
        "python analyze_saved_predictions.py --target ge_c --model xgboost",
        "python analyze_saved_predictions.py --target ge_c --model logistic_regression",
        "python analyze_saved_predictions.py --target ge_c --model histgradientboosting",
        "python analyze_saved_predictions.py --target ge_c --model extratrees",
        "python analyze_saved_predictions.py --target ge_m --model xgboost",
        "python analyze_saved_predictions.py --target ge_m --model logistic_regression",
        "python analyze_saved_predictions.py --target ge_m --model histgradientboosting",
        "python analyze_saved_predictions.py --target ge_m --model extratrees",
        "python research_analysis.py",
        "```",
        "",
    ])
    (ROOT / "RESULTS_FINAL.md").write_text("\n".join(lines))


def main() -> None:
    summary = _metrics_summary()
    _paired_auc_comparisons()
    _roc_pr_figures()
    _calibration()
    _threshold_sensitivity()
    _dotwhiskers(summary)
    _baseline_comparison()
    _cnn_lstm_artifacts()
    _workflow_figure()
    _literature_table()
    _feature_importance()
    _write_final_report(summary)
    print("Saved paired comparisons, calibration, threshold sensitivity, baselines, and tabular figures from saved predictions")


if __name__ == "__main__":
    main()
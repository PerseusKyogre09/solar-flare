"""Resumable publication metrics from one saved binary prediction pair."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.research_metrics import (
    BINARY_BOOTSTRAP_METRICS,
    cluster_bootstrap_metric_samples,
    make_cluster_bootstrap_counts,
    percentile_intervals,
)
from src.xgb_baseline.metrics import comprehensive_binary_metrics, select_validation_threshold


ROOT = Path(__file__).resolve().parent
N_RESAMPLES = 2000
SEED = 42
MODEL_NAMES = {
    "xgboost": "XGBoost",
    "logistic_regression": "Logistic Regression",
    "histgradientboosting": "HistGradientBoosting",
    "extratrees": "Extra Trees",
}
PREDICTION_SOURCES = {
    "xgboost": ("xgboost", "validation_xgboost.csv", "test_xgboost.csv"),
    "logistic_regression": (
        "xgboost",
        "validation_logistic_regression.csv",
        "test_logistic_regression.csv",
    ),
    "histgradientboosting": ("histgb", "validation_predictions.csv", "test_predictions.csv"),
    "extratrees": ("extratrees", "validation_predictions.csv", "test_predictions.csv"),
}
OUTPUT_FAMILIES = {
    "xgboost": "xgboost",
    "logistic_regression": "logistic_regression",
    "histgradientboosting": "histgb",
    "extratrees": "extratrees",
}
REQUIRED_COLUMNS = {"row_id", "active_region", "target", "probability"}
SUMMARY_METRICS = (
    "roc_auc", "pr_auc", "tss", "hss", "csi", "mcc", "f1", "precision",
    "pod_recall_tpr", "specificity", "far", "brier_score",
)


def _load_pair(model: str, target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    family, validation_name, test_name = PREDICTION_SOURCES[model]
    base = ROOT / "experiments" / family / target / "predictions"
    validation = pd.read_csv(base / validation_name)
    test = pd.read_csv(base / test_name)
    for name, frame in (("validation", validation), ("test", test)):
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"{model}/{target} {name} predictions lack columns: {sorted(missing)}")
        if frame.row_id.duplicated().any():
            raise ValueError(f"Duplicate row_id in {model}/{target} {name} predictions")
        if frame.target.isna().any() or not frame.target.isin([0, 1]).all():
            raise ValueError(f"Invalid labels in {model}/{target} {name} predictions")
        if not np.isfinite(frame.probability.to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite probabilities in {model}/{target} {name} predictions")
    return validation.sort_values("row_id").reset_index(drop=True), test.sort_values("row_id").reset_index(drop=True)


def _check_split_alignment(model: str, target: str, validation: pd.DataFrame, test: pd.DataFrame) -> None:
    if model == "xgboost":
        return
    reference_validation = pd.read_csv(
        ROOT / "experiments/xgboost" / target / "predictions/validation_xgboost.csv",
        usecols=["row_id", "active_region", "target"],
    ).sort_values("row_id").reset_index(drop=True)
    reference_test = pd.read_csv(
        ROOT / "experiments/xgboost" / target / "predictions/test_xgboost.csv",
        usecols=["row_id", "active_region", "target"],
    ).sort_values("row_id").reset_index(drop=True)
    for name, candidate, reference in (
        ("validation", validation, reference_validation),
        ("test", test, reference_test),
    ):
        if len(candidate) != len(reference):
            raise ValueError(f"{model}/{target} {name} row count differs from XGBoost")
        if not candidate.row_id.equals(reference.row_id):
            raise ValueError(f"{model}/{target} {name} row IDs do not match XGBoost")
        if not candidate.target.equals(reference.target):
            raise ValueError(f"{model}/{target} {name} labels do not match XGBoost")
        if not candidate.active_region.astype(str).equals(reference.active_region.astype(str)):
            raise ValueError(f"{model}/{target} {name} active-region IDs do not match XGBoost")


def _write_prediction(frame: pd.DataFrame, threshold: float, path: Path) -> None:
    prediction = frame.copy()
    prediction["predicted_label"] = (prediction.probability >= threshold).astype("int8")
    prediction["selected_threshold"] = threshold
    path.parent.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(path, index=False)


def _update_results_csv(model: str, target: str, payload: dict, intervals: dict) -> None:
    results_path = ROOT / "results/results_final.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()
    old = old.loc[~((old.get("model", pd.Series(dtype=str)) == MODEL_NAMES[model]) &
                    (old.get("target", pd.Series(dtype=str)) == target))]
    row = {
        "model": MODEL_NAMES[model],
        "target": target,
        "threshold": payload["selected_threshold"],
        "positive_prevalence": payload["test"]["positive_prevalence"],
        **{key: payload["test"][key] for key in SUMMARY_METRICS},
        **{f"{key}_ci_low": intervals[key][0] for key in SUMMARY_METRICS},
        **{f"{key}_ci_high": intervals[key][1] for key in SUMMARY_METRICS},
    }
    pd.concat([old, pd.DataFrame([row])], ignore_index=True).sort_values(["target", "model"]).to_csv(
        results_path, index=False
    )


def run_one(model: str, target: str) -> Path:
    validation, test = _load_pair(model, target)
    _check_split_alignment(model, target, validation, test)
    thresholds = json.loads((ROOT / "src/xgb_baseline/config.json").read_text())["thresholds"]
    threshold = select_validation_threshold(validation.target, validation.probability, thresholds)
    validation_metrics = comprehensive_binary_metrics(validation.target, validation.probability, threshold)
    test_metrics = comprehensive_binary_metrics(test.target, test.probability, threshold)
    output_family = OUTPUT_FAMILIES[model]
    model_dir = ROOT / "experiments" / output_family / target
    prediction_dir = model_dir / "predictions"
    _write_prediction(validation, threshold, prediction_dir / "validation_predictions.csv")
    test_predictions_path = prediction_dir / "test_predictions.csv"
    _write_prediction(test, threshold, test_predictions_path)

    payload = {
        "model": model,
        "target": target,
        "selected_threshold": threshold,
        "threshold_selection": {
            "split": "Validation",
            "metric": "TSS",
            "candidates": thresholds,
            "validation_tss": validation_metrics["tss"],
        },
        "validation": validation_metrics,
        "test": test_metrics,
        "test_unit": "observation",
        "confidence_intervals": None,
    }
    metrics_path = model_dir / "metrics/comprehensive_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")

    if test.active_region.isna().any():
        raise ValueError(f"{model}/{target} has missing active-region IDs; refusing observation bootstrap")
    _, row_codes, cluster_counts = make_cluster_bootstrap_counts(
        test.active_region.astype(str).to_numpy(), n_resamples=N_RESAMPLES, seed=SEED
    )
    samples = cluster_bootstrap_metric_samples(
        test.target.to_numpy(),
        test.probability.to_numpy(),
        row_codes,
        cluster_counts,
        threshold,
    )
    intervals = percentile_intervals(samples)
    payload["confidence_intervals"] = intervals
    payload["bootstrap"] = {
        "resamples": N_RESAMPLES,
        "seed": SEED,
        "method": "paired percentile cluster bootstrap resampling active regions with replacement",
        "cluster_count": int(test.active_region.nunique()),
    }
    metrics_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    _update_results_csv(model, target, payload, intervals)

    uncertainty_dir = ROOT / "results/statistical_uncertainty"
    uncertainty_dir.mkdir(parents=True, exist_ok=True)
    uncertainty_path = uncertainty_dir / f"{model}_{target}_bootstrap.json"
    uncertainty_path.write_text(json.dumps({"model": model, "target": target, **payload["bootstrap"], "ci_95": intervals}, indent=2) + "\n")
    print(
        f"{model}/{target}: n={len(test)} regions={test.active_region.nunique()} "
        f"threshold={threshold:.1f} ROC-AUC={test_metrics['roc_auc']:.6f} "
        f"TSS={test_metrics['tss']:.6f} CI={intervals['tss']}"
    )
    return metrics_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("ge_c", "ge_m"), required=True)
    parser.add_argument("--model", choices=tuple(MODEL_NAMES), required=True)
    args = parser.parse_args()
    run_one(args.model, args.target)


if __name__ == "__main__":
    main()

"""Build missing validation/test probabilities from already-fitted tree models."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.xgb_baseline.data import parse_filename, target_from_flare
from src.xgb_baseline.metrics import comprehensive_binary_metrics, select_validation_threshold


ROOT = Path(__file__).resolve().parent
FAMILIES = {
    "histgradientboosting": ("histgb", "HistGradientBoostingClassifier"),
    "extratrees": ("extratrees", "ExtraTreesClassifier"),
}
FEATURE_NAMES = [f"feature_{index:02d}" for index in range(29)]
CHUNK_ROWS = 50_000


def _basename(value: str) -> str:
    return str(value).replace("\\", "/").rsplit("/", 1)[-1]


def _split_metadata(target: str) -> tuple[dict[str, dict], dict[str, str]]:
    metadata = {}
    wanted = {}
    for split in ("Validation", "Test"):
        path = ROOT / "data" / f"{split}_Data_by_AR_png_224.csv"
        frame = pd.read_csv(path, usecols=["filename"])
        active_region, timestamp = parse_filename(frame.filename)
        if active_region.isna().any() or timestamp.isna().any():
            raise ValueError(f"Malformed active-region/timestamp in {path}")
        for filename, region, time in zip(frame.filename, active_region, timestamp):
            row_id = _basename(filename)
            if row_id in metadata:
                raise ValueError(f"Duplicate split filename: {row_id}")
            metadata[row_id] = {
                "filename": str(filename),
                "active_region": str(region),
                "timestamp": time,
                "split": split,
            }
            wanted[row_id] = split
    return metadata, wanted


def _read_selected_labels(wanted: set[str]) -> dict[str, str]:
    labels = {}
    with (ROOT / "data/C1.0_24hr_224_png_Labels.txt").open(newline="") as handle:
        for filename, flare in csv.reader(handle):
            row_id = _basename(filename)
            if row_id in wanted:
                labels[row_id] = flare.strip().upper()
    missing = wanted - labels.keys()
    if missing:
        raise ValueError(f"Missing {len(missing)} validation/test labels; first={next(iter(missing))}")
    return labels


def _extract_validation_test(metadata: dict[str, dict], wanted: dict[str, str]) -> dict[str, pd.DataFrame]:
    selected = {"Validation": [], "Test": []}
    seen = set()
    feature_path = ROOT / "data/Lat60_Lon60_Nans0_C1.0_24hr_png_224_features.csv"
    usecols = [*range(29), 31]
    for chunk in pd.read_csv(
        feature_path,
        header=None,
        usecols=usecols,
        chunksize=CHUNK_ROWS,
        low_memory=False,
    ):
        names = chunk[31].astype("string").map(_basename)
        keep = names.isin(wanted)
        if not keep.any():
            continue
        part = chunk.loc[keep, [*range(29)]].copy()
        part.columns = FEATURE_NAMES
        part = part.apply(pd.to_numeric, errors="coerce")
        part.insert(0, "row_id", names.loc[keep].to_numpy())
        duplicates = set(part.row_id) & seen
        if duplicates:
            raise ValueError(f"Duplicate feature filename: {next(iter(duplicates))}")
        seen.update(part.row_id)
        for split, rows in part.groupby(part.row_id.map(wanted.__getitem__), sort=False):
            selected[split].append(rows.copy())

    missing = set(wanted) - seen
    if missing:
        raise ValueError(f"Missing {len(missing)} validation/test feature rows; first={next(iter(missing))}")

    result = {}
    for split, parts in selected.items():
        frame = pd.concat(parts, ignore_index=True)
        details = pd.DataFrame.from_dict(
            {row_id: metadata[row_id] for row_id in frame.row_id},
            orient="index",
        ).rename_axis("row_id").reset_index()
        frame = frame.merge(details, on="row_id", how="left", validate="one_to_one", sort=False)
        result[split] = frame
    return result


def _check_saved_metrics(metrics_path: Path, threshold: float, test_metrics: dict) -> None:
    saved = json.loads(metrics_path.read_text())
    saved_threshold = float(saved["selected_threshold"])
    if not np.isclose(threshold, saved_threshold, rtol=0, atol=1e-12):
        raise ValueError(f"Validation-selected threshold {threshold} disagrees with saved {saved_threshold}")
    keys = {
        "roc_auc": "roc_auc",
        "pr_auc": "pr_auc",
        "tss": "tss",
        "f1": "f1",
        "precision": "precision",
        "recall": "pod_recall_tpr",
    }
    for saved_key, metric_key in keys.items():
        if not np.isclose(test_metrics[metric_key], saved["test"][saved_key], rtol=1e-6, atol=1e-7):
            raise ValueError(
                f"Recomputed test {saved_key}={test_metrics[metric_key]} "
                f"disagrees with saved {saved['test'][saved_key]}"
            )


def build(target: str, family: str) -> Path:
    experiment_family, model_class = FAMILIES[family]
    model_dir = ROOT / "experiments" / experiment_family / target
    model = joblib.load(model_dir / "model.joblib")
    if model.__class__.__name__ != model_class or model.n_features_in_ != len(FEATURE_NAMES):
        raise ValueError(f"Unexpected model artifact: {type(model).__name__}")

    metadata, wanted = _split_metadata(target)
    labels = _read_selected_labels(set(wanted))
    frames = _extract_validation_test(metadata, wanted)
    for frame in frames.values():
        frame["flare"] = frame.row_id.map(labels)
        frame["target"] = target_from_flare(frame.flare, target).astype("int8")

    validation = frames["Validation"]
    test = frames["Test"]
    thresholds = json.loads((ROOT / "src/xgb_baseline/config.json").read_text())["thresholds"]
    validation_probability = model.predict_proba(validation[FEATURE_NAMES])[:, 1]
    threshold = select_validation_threshold(validation.target, validation_probability, thresholds)
    test_probability = model.predict_proba(test[FEATURE_NAMES])[:, 1]
    validation_metrics = comprehensive_binary_metrics(validation.target, validation_probability, threshold)
    test_metrics = comprehensive_binary_metrics(test.target, test_probability, threshold)
    _check_saved_metrics(model_dir / "metrics.json", threshold, test_metrics)

    output_dir = model_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for split, frame, probability in (
        ("validation", validation, validation_probability),
        ("test", test, test_probability),
    ):
        prediction = frame[["row_id", "filename", "active_region", "timestamp", "flare", "target"]].copy()
        prediction["split"] = split.title()
        prediction["probability"] = probability
        prediction["predicted_label"] = (probability >= threshold).astype("int8")
        prediction["selected_threshold"] = threshold
        outputs.append((output_dir / f"{split}_predictions.csv", prediction))
    for path, prediction in outputs:
        prediction.to_csv(path, index=False)

    payload = {
        "model": family,
        "model_class": model_class,
        "target": target,
        "selected_threshold": threshold,
        "validation_tss": validation_metrics["tss"],
        "test": test_metrics,
        "prediction_source": "existing fitted model applied to chunk-extracted validation/test features",
    }
    metrics_path = output_dir.parent / "metrics" / "comprehensive_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(
        f"{family}/{target}: validation={len(validation)} test={len(test)} "
        f"threshold={threshold:.1f} TSS={test_metrics['tss']:.6f}"
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("ge_c", "ge_m"), required=True)
    parser.add_argument("--model", choices=tuple(FAMILIES), required=True)
    args = parser.parse_args()
    build(args.target, args.model)


if __name__ == "__main__":
    main()
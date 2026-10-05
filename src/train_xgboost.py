from __future__ import annotations

import argparse
import inspect
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .xgb_baseline.data import DataValidationError, load_dataset
from .xgb_baseline.metrics import binary_metrics, select_validation_threshold, threshold_table


def _json_safe(value):
    if isinstance(value, dict): return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return None if not np.isfinite(value) else float(value)
    return value


def write_json(path: Path, value):
    path.write_text(json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n")


def _weights(y: pd.Series, mode: str) -> np.ndarray | None:
    if mode == "none": return None
    if mode != "balanced": raise ValueError("class_weighting must be none or balanced")
    pos, neg = int(y.sum()), int((y == 0).sum())
    if not pos or not neg: raise DataValidationError("Cannot calculate balanced weights for a one-class training split")
    return np.where(y.to_numpy() == 1, neg / pos, 1.0)


def _region_metrics(frame: pd.DataFrame, probability: np.ndarray, threshold: float) -> dict:
    grouped = frame.assign(probability=probability).groupby("active_region", sort=True)
    region = grouped.agg(target=("target", "max"), probability=("probability", "max"))
    result = binary_metrics(region.target, region.probability, threshold)
    result["unit"] = "active_region"
    result["aggregation"] = "maximum probability; maximum observed target; thresholded any-region rule"
    result["active_region_count"] = int(len(region))
    return result


def _limit_stratified(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Create a deterministic smoke subset while retaining available classes."""
    if len(frame) <= limit:
        return frame
    parts = [part.head(max(1, int(round(limit * len(part) / len(frame))))) for _, part in frame.groupby("target", sort=True)]
    result = pd.concat(parts).sort_values(["active_region", "timestamp"])
    if len(result) > limit:
        result = result.head(limit)
    return result


def _evaluate(name: str, frame: pd.DataFrame, model, features: list[str], thresholds: list[float], selected: float, out: Path) -> dict:
    probability = model.predict_proba(frame[features])[:, 1]
    predictions = pd.DataFrame({"row_id": frame.row_id, "filename": frame.feature_filename, "active_region": frame.active_region, "timestamp": frame.timestamp, "split": frame.split, "flare": frame.flare, "target": frame.target, "probability": probability})
    predictions.to_csv(out / "predictions" / f"{name}.csv", index=False)
    obs = threshold_table(frame.target, probability, sorted(set([0.5, selected] + thresholds)))
    for row in obs: row["unit"] = "observation"; row["split"] = name
    rows = obs + [_region_metrics(frame, probability, selected) | {"split": name}]
    return {"threshold_metrics": obs, "selected_threshold": selected, "active_region_selected": rows[-1]}


def run(args) -> Path:
    config = json.loads(Path(args.config).read_text())
    target = args.target or config.get("target", "ge_c")
    if args.features: config["features"] = args.features
    bundle = load_dataset(config["features"], config["labels"], config["split_dir"], target)
    df = bundle.frame
    if args.limit_per_split:
        df = pd.concat([_limit_stratified(part.sort_values(["active_region", "timestamp"]), args.limit_per_split) for _, part in df.groupby("split", sort=False)], ignore_index=True)
    out = Path(config.get("output_dir", "experiments/xgboost")) / target
    for sub in ("configs", "models", "metrics", "predictions", "reports", "feature_importance", "logs"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 42)); np.random.seed(seed)
    write_json(out / "configs" / "run.json", config | {"target": target, "resolved_at_utc": datetime.now(timezone.utc).isoformat(), "feature_names": bundle.feature_names})
    write_json(out / "reports" / "dataset_summary.json", bundle.report)
    (out / "configs" / "feature_names.txt").write_text("\n".join(bundle.feature_names) + "\n")
    try:
        env = subprocess.check_output([str(Path("venv/bin/python")), "-m", "pip", "list", "--format=freeze"], text=True, stderr=subprocess.STDOUT)
    except Exception:
        env = platform.platform() + "\n" + platform.python_version() + "\n"
    (out / "reports" / "environment.txt").write_text(env)
    feature_cols = bundle.feature_names
    train, val, test = (df[df.split.eq(s)].copy() for s in ("Train", "Validation", "Test"))
    X_train, X_val, X_test = (x[feature_cols] for x in (train, val, test)); y_train, y_val, y_test = (x.target for x in (train, val, test))
    if config.get("impute_training_median", False):
        imputer = SimpleImputer(strategy="median"); X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_cols, index=X_train.index); X_val = pd.DataFrame(imputer.transform(X_val), columns=feature_cols, index=X_val.index); X_test = pd.DataFrame(imputer.transform(X_test), columns=feature_cols, index=X_test.index); joblib.dump(imputer, out / "models" / "imputer.joblib")
    weights = _weights(y_train, config.get("class_weighting", "none"))
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError("xgboost is not installed in the active environment. Install a CPU-compatible xgboost build, then rerun.") from exc
    params = dict(config.get("xgboost", {})); params.setdefault("random_state", seed); params.setdefault("n_jobs", 1)
    early_stopping = int(config.get("early_stopping_rounds", 40))
    constructor_supports_early = "early_stopping_rounds" in inspect.signature(XGBClassifier).parameters
    if constructor_supports_early:
        xgb = XGBClassifier(**params, early_stopping_rounds=early_stopping)
    else:
        xgb = XGBClassifier(**params)
    fit_kwargs = {"eval_set": [(X_train, y_train), (X_val, y_val)], "verbose": False}
    if weights is not None: fit_kwargs["sample_weight"] = weights
    try:
        if constructor_supports_early:
            xgb.fit(X_train, y_train, **fit_kwargs)
        else:
            xgb.fit(X_train, y_train, early_stopping_rounds=early_stopping, **fit_kwargs)
    except TypeError:
        xgb.fit(X_train, y_train, **fit_kwargs)
    xgb.save_model(out / "models" / "xgboost.json")
    write_json(out / "models" / "xgboost_config.json", {"params": params, "class_weighting": config.get("class_weighting", "none"), "feature_names": feature_cols})
    importance = pd.DataFrame({"feature": feature_cols, "gain": [xgb.get_booster().get_score(importance_type="gain").get(f"f{i}", 0.0) for i in range(len(feature_cols))], "weight_frequency": [xgb.get_booster().get_score(importance_type="weight").get(f"f{i}", 0.0) for i in range(len(feature_cols))]}).sort_values("gain", ascending=False)
    importance.to_csv(out / "feature_importance" / "xgboost_importance.csv", index=False)
    val_probability = xgb.predict_proba(X_val)[:, 1]
    thresholds = [float(t) for t in config.get("thresholds", [0.5])]
    selected = select_validation_threshold(y_val, val_probability, thresholds)
    results = {"target": target, "selected_threshold": selected, "models": {}}
    # All models are evaluated using the same untouched validation/test rows.
    class_prior = float(y_train.mean())
    class_prior_model = type("PriorModel", (), {"predict_proba": lambda self, X: np.column_stack([np.full(len(X), 1 - class_prior), np.full(len(X), class_prior)])})()
    models = {"majority_climatology": class_prior_model, "xgboost": xgb}
    logit = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", LogisticRegression(max_iter=500, class_weight="balanced" if config.get("class_weighting") == "balanced" else None, random_state=seed))])
    logit.fit(X_train, y_train); joblib.dump(logit, out / "models" / "logistic_regression.joblib"); models["logistic_regression"] = logit
    for model_name, model in models.items():
        model_out = out / "metrics" / f"{model_name}.json"
        val_res = _evaluate(f"validation_{model_name}", val, model, feature_cols, thresholds, selected, out)
        test_res = _evaluate(f"test_{model_name}", test, model, feature_cols, thresholds, selected, out)
        results["models"][model_name] = {"validation": val_res, "test": test_res}
        write_json(model_out, results["models"][model_name])
    write_json(out / "metrics" / "all_results.json", results)
    report = f"""# XGBoost Solar-Flare Baseline\n\n- Target: `{target}` (`0` is B/no qualifying flare; positive is {', '.join(sorted(__import__('src.xgb_baseline.data', fromlist=['TARGETS']).TARGETS[target]))}).\n- Features: {len(feature_cols)} Dryad engineered feature columns. These are not claimed to be identical to DeFN's 79 features.\n- Splits: predefined active-region grouped Train/Validation/Test; overlap checks passed. Timestamp ranges overlap, so this is grouped—not chronological—evaluation.\n- Threshold: selected on validation by maximum TSS over `{thresholds}`; test is evaluated once at that threshold and the configured thresholds.\n- Imputation: {'training-median, fitted on Train only' if config.get('impute_training_median') else 'disabled; XGBoost native missing-value handling is used; logistic regression has its own training-only median imputer'}.\n- Region summary: maximum predicted probability and maximum observed target per active region. This is a region-level operating summary, not an independent-event estimate.\n\nSee `dataset_summary.json`, `all_results.json`, prediction CSVs, and feature importance CSV for machine-readable results. Gain and frequency importance are model diagnostics, not causal evidence.\n\n## Limitations\n\nThe required feature CSV was not present in the repository at implementation time; the configured path must point to the actual Dryad 29-feature file. No full experiment is run until that file is supplied and xgboost is installed.\n"""
    (out / "reports" / "baseline.md").write_text(report)
    return out


def main():
    parser = argparse.ArgumentParser(description="Train the CPU XGBoost 29-feature solar-flare baseline")
    parser.add_argument("--target", choices=["ge_c", "ge_m"]); parser.add_argument("--config", default="src/xgb_baseline/config.json"); parser.add_argument("--features", help="Override configured Dryad 29-feature CSV")
    parser.add_argument("--limit-per-split", type=int, help="Small smoke-test limit after validation/alignment")
    args = parser.parse_args()
    try: print(f"Outputs: {run(args)}")
    except (FileNotFoundError, DataValidationError, RuntimeError, ValueError) as exc: parser.error(str(exc))


if __name__ == "__main__": main()

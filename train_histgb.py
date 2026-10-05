"""Train a CPU-friendly third tabular baseline on the existing aligned dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.xgb_baseline.data import load_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", choices=["ge_c", "ge_m"], required=True)
    p.add_argument("--config", default="src/xgb_baseline/config.json")
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    bundle = load_dataset(cfg["features"], cfg["labels"], cfg["split_dir"], args.target)
    df, cols = bundle.frame, bundle.feature_names
    train, val, test = (df[df.split.eq(x)] for x in ("Train", "Validation", "Test"))
    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
        l2_regularization=1.0, random_state=42, early_stopping=True,
    )
    model.fit(train[cols], train.target)
    val_p = model.predict_proba(val[cols])[:, 1]
    thresholds = np.arange(0.1, 1.0, 0.1)
    best_t = max(thresholds, key=lambda t: _tss(val.target.to_numpy(), val_p, t))
    test_p = model.predict_proba(test[cols])[:, 1]
    result = {"target": args.target, "selected_threshold": float(best_t), "model": "HistGradientBoostingClassifier", "test": _metrics(test.target.to_numpy(), test_p, best_t)}
    out = Path("experiments/histgb") / args.target
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    model_path = out / "model.joblib"
    import joblib
    joblib.dump(model, model_path)
    print(json.dumps(result, indent=2))


def _tss(y, p, t):
    pred = p >= t
    pos, neg = y == 1, y == 0
    return float((pred[pos].mean() if pos.any() else 0) - (pred[neg].mean() if neg.any() else 0))


def _metrics(y, p, t):
    pred = p >= t
    return {"roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)), "tss": _tss(y, p, t), "f1": float(f1_score(y, pred, zero_division=0)), "precision": float(precision_score(y, pred, zero_division=0)), "recall": float(recall_score(y, pred, zero_division=0)), "balanced_accuracy": float(balanced_accuracy_score(y, pred))}


if __name__ == "__main__":
    main()

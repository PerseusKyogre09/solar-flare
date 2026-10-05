"""Train an Extra-Trees tabular baseline on the validated dataset."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from src.xgb_baseline.data import load_dataset

def tss(y, p, threshold):
    pred = p >= threshold
    return float(pred[y == 1].mean() - pred[y == 0].mean())

def metrics(y, p, threshold):
    pred = p >= threshold
    return {"roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)), "tss": tss(y, p, threshold), "f1": float(f1_score(y, pred, zero_division=0)), "precision": float(precision_score(y, pred, zero_division=0)), "recall": float(recall_score(y, pred, zero_division=0)), "balanced_accuracy": float(balanced_accuracy_score(y, pred))}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["ge_c", "ge_m"], required=True)
    ap.add_argument("--config", default="src/xgb_baseline/config.json")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    bundle = load_dataset(cfg["features"], cfg["labels"], cfg["split_dir"], args.target)
    df, cols = bundle.frame, bundle.feature_names
    train, val, test = (df[df.split.eq(s)] for s in ("Train", "Validation", "Test"))
    model = ExtraTreesClassifier(n_estimators=160, max_features="sqrt", min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=42)
    model.fit(train[cols], train.target)
    vp = model.predict_proba(val[cols])[:, 1]
    thresholds = np.arange(0.1, 1.0, 0.1)
    selected = max(thresholds, key=lambda x: tss(val.target.to_numpy(), vp, x))
    tp = model.predict_proba(test[cols])[:, 1]
    result = {"target": args.target, "model": "ExtraTreesClassifier", "selected_threshold": float(selected), "validation": {"tss": tss(val.target.to_numpy(), vp, selected)}, "test": metrics(test.target.to_numpy(), tp, selected)}
    out = Path("experiments/extratrees") / args.target
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    joblib.dump(model, out / "model.joblib")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()

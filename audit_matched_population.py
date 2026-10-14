"""Read-only audit of sequence eligibility and matched sequence endpoints.

Writes only new artifacts below results/matched_audit/. It never rewrites
sequence manifests, model outputs, or historical results.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "matched_audit"
LABEL_PATH = ROOT / "data" / "C1.0_24hr_224_png_Labels.txt"
SEQUENCE_LENGTH = 10
STRIDE = 10
CADENCE_MINUTES = 12


def load_labels() -> dict[str, str]:
    labels: dict[str, str] = {}
    with LABEL_PATH.open() as handle:
        for line in handle:
            filename, flare = line.rstrip("\n").split(",", 1)
            labels[filename] = flare
    return labels


def class_of(flare: object) -> str | None:
    text = str(flare) if pd.notna(flare) else ""
    return text[0] if text[:1] in {"C", "M", "X"} else None


def parse_time(names: pd.Series) -> pd.Series:
    stamp = names.str.extract(r"\.(\d{8}_\d{6})_TAI", expand=False)
    return pd.to_datetime(stamp, format="%Y%m%d_%H%M%S", errors="coerce")


def split_rows(split: str, labels: dict[str, str]) -> tuple[pd.DataFrame, list[dict]]:
    source = pd.read_csv(ROOT / "data" / f"{split}_Data_by_AR_png_224.csv", dtype="string")
    source["ar"] = source["filename"].str.split("/", n=1).str[0]
    source["name"] = source["filename"].str.split("/", n=1).str[1]
    source["flare"] = source["name"].map(labels)
    source["class"] = source["flare"].map(class_of)
    source["timestamp"] = parse_time(source["name"])

    ledger: list[dict] = []
    sequence_rows: list[dict] = []
    for ar, group in source.groupby("ar", sort=True):
        group = group.sort_values("timestamp", kind="mergesort")
        valid = group.dropna(subset=["class", "timestamp"]).copy()
        candidate_windows = 0
        valid_windows = 0
        cadence_failures = 0
        missing_image_windows = 0
        last_reason = "no_valid_CMX_timestamp_rows"
        rows = valid.to_dict("records")
        for start in range(0, max(0, len(rows) - SEQUENCE_LENGTH + 1), STRIDE):
            candidate_windows += 1
            window = rows[start : start + SEQUENCE_LENGTH]
            times = [row["timestamp"] for row in window]
            cadence_ok = all(
                (times[i + 1] - times[i]).total_seconds() / 60 == CADENCE_MINUTES
                for i in range(SEQUENCE_LENGTH - 1)
            )
            if not cadence_ok:
                cadence_failures += 1
                continue
            valid_windows += 1
            last_reason = "eligible"
            endpoint = window[-1]
            sequence_rows.append(
                {
                    "split": split,
                    "active_region": str(ar),
                    "endpoint_filename": endpoint["filename"],
                    "endpoint_basename": endpoint["name"],
                    "endpoint_timestamp": endpoint["timestamp"].strftime("%Y-%m-%dT%H:%M:%S"),
                    "target_class": endpoint["class"],
                    "target_ge_m": int(endpoint["class"] in {"M", "X"}),
                    "frames": json.dumps([row["filename"] for row in window]),
                }
            )
        if len(valid) == 0:
            reason = "no_valid_CMX_timestamp_rows"
        elif candidate_windows == 0:
            reason = "fewer_than_10_valid_CMX_timestamp_rows"
        elif valid_windows == 0:
            reason = "all_candidate_windows_failed_12_minute_cadence"
        else:
            reason = "eligible"
        ledger.append(
            {
                "split": split,
                "active_region": str(ar),
                "raw_rows": int(len(group)),
                "unmatched_labels": int(group["flare"].isna().sum()),
                "invalid_timestamps": int(group["timestamp"].isna().sum()),
                "zero_label_rows": int((group["flare"] == "0").sum()),
                "C_rows": int((group["class"] == "C").sum()),
                "M_rows": int((group["class"] == "M").sum()),
                "X_rows": int((group["class"] == "X").sum()),
                "valid_CMX_rows": int(len(valid)),
                "candidate_windows": candidate_windows,
                "cadence_failures": cadence_failures,
                "valid_windows": valid_windows,
                "eligibility_reason": reason,
            }
        )
    return pd.DataFrame(sequence_rows), ledger


def tabular_endpoint_index(labels: dict[str, str]) -> pd.DataFrame:
    parts = []
    for split in ("Train", "Validation", "Test"):
        frame = pd.read_csv(ROOT / "data" / f"{split}_Data_by_AR_png_224.csv", dtype="string")
        frame["split"] = split
        frame["active_region"] = frame["filename"].str.split("/", n=1).str[0]
        frame["basename"] = frame["filename"].str.split("/", n=1).str[1]
        frame["timestamp"] = parse_time(frame["basename"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
        frame["flare"] = frame["basename"].map(labels)
        frame["target_ge_m"] = frame["flare"].map(
            lambda value: int(isinstance(value, str) and value[:1] in {"M", "X"})
        ).astype("int8")
        parts.append(frame[["split", "active_region", "filename", "basename", "timestamp", "target_ge_m"]])
    return pd.concat(parts, ignore_index=True)


def seed_inventory() -> pd.DataFrame:
    requested = [11, 12, 34, 42, 56, 69, 90, 93, 104, 702, 1000]
    rows = []
    for seed in requested:
        directory = ROOT / "runs" / "cnn_lstm" / f"seed_{seed}"
        expected = {
            "best_checkpoint": directory / f"cnn_lstm_best_seed_{seed}.pth",
            "final_checkpoint": directory / f"cnn_lstm_final_seed_{seed}.pth",
            "history": directory / f"cnn_lstm_training_history_seed_{seed}.csv",
            "config": directory / f"cnn_lstm_config_seed_{seed}.txt",
            "test_json": directory / "cnn_lstm_test_results.json",
            "classification_report": directory / "cnn_lstm_classification_report.txt",
            "confusion_matrix": directory / "cnn_lstm_confusion_matrix.png",
        }
        rows.append(
            {
                "seed": seed,
                "directory": str(directory.relative_to(ROOT)) if directory.exists() else "",
                **{name: int(path.exists()) for name, path in expected.items()},
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    labels = load_labels()
    ledgers = []
    endpoints = []
    for split in ("Train", "Validation", "Test"):
        seq, ledger = split_rows(split, labels)
        endpoints.append(seq)
        ledgers.extend(ledger)
    eligibility = pd.DataFrame(ledgers)
    endpoint_frame = pd.concat(endpoints, ignore_index=True)
    tabular = tabular_endpoint_index(labels)
    matched = endpoint_frame.merge(
        tabular,
        left_on=["split", "active_region", "endpoint_basename", "endpoint_timestamp"],
        right_on=["split", "active_region", "basename", "timestamp"],
        how="outer",
        indicator=True,
        suffixes=("_sequence", "_tabular"),
    )
    matched["target_agrees"] = matched["target_ge_m_sequence"] == matched["target_ge_m_tabular"]
    matched.to_csv(OUT / "sequence_endpoint_tabular_match.csv", index=False)
    eligibility.to_csv(OUT / "sequence_eligibility_ledger.csv", index=False)
    seed_inventory().to_csv(OUT / "cnn_seed_inventory.csv", index=False)
    summary = {
        "requested_seeds": [11, 12, 34, 42, 56, 69, 90, 93, 104, 702, 1000],
        "sequence_counts": {
            split: {
                "raw_rows": int((eligibility.split == split).sum() and eligibility.loc[eligibility.split == split, "raw_rows"].sum()),
                "raw_active_regions": int(eligibility.loc[eligibility.split == split, "active_region"].nunique()),
                "valid_sequences": int((endpoint_frame.split == split).sum()),
                "sequence_active_regions": int(endpoint_frame.loc[endpoint_frame.split == split, "active_region"].nunique()),
            }
            for split in ("Train", "Validation", "Test")
        },
        "matching": {
            "total_sequence_endpoints": int(len(endpoint_frame)),
            "both": int((matched["_merge"] == "both").sum()),
            "sequence_only": int((matched["_merge"] == "left_only").sum()),
            "tabular_only": int((matched["_merge"] == "right_only").sum()),
            "target_disagreements": int(((matched["_merge"] == "both") & ~matched["target_agrees"]).sum()),
            "matched_ge_m_positive": int(matched.loc[matched["_merge"] == "both", "target_ge_m_sequence"].sum()),
            "matched_ge_m_negative": int(((matched["_merge"] == "both") & (matched["target_ge_m_sequence"] == 0)).sum()),
        },
    }
    (OUT / "matched_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()

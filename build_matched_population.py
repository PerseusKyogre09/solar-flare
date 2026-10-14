"""Build and validate the all-label matched >=M endpoint population.

This script writes only new artifacts under results/matched_experiment/.
It does not modify the existing sequence manifests or historical results.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "matched_experiment"
LABELS = ROOT / "data" / "C1.0_24hr_224_png_Labels.txt"
IMAGE_ROOT = ROOT / "data" / "images" / "Lat60_Lon60_Nans0_png_224"
SEQUENCE_LENGTH = 10
STRIDE = 10
CADENCE_MINUTES = 12


def load_labels() -> dict[str, str]:
    result = {}
    with LABELS.open() as handle:
        for line in handle:
            name, label = line.rstrip("\n").split(",", 1)
            result[name] = label
    return result


def parse_times(names: pd.Series) -> pd.Series:
    stamp = names.str.extract(r"\.(\d{8}_\d{6})_TAI", expand=False)
    return pd.to_datetime(stamp, format="%Y%m%d_%H%M%S", errors="coerce")


def make_split(split: str, labels: dict[str, str]) -> tuple[pd.DataFrame, dict]:
    source = pd.read_csv(ROOT / "data" / f"{split}_Data_by_AR_png_224.csv", dtype="string")
    source["ar"] = source.filename.str.split("/", n=1).str[0]
    source["basename"] = source.filename.str.split("/", n=1).str[1]
    source["flare_label"] = source.basename.map(labels)
    source["timestamp"] = parse_times(source.basename)
    source = source.dropna(subset=["flare_label", "timestamp"])

    rows = []
    candidate_windows = 0
    cadence_failures = 0
    for ar, group in source.groupby("ar", sort=True):
        ordered = group.sort_values("timestamp", kind="mergesort").to_dict("records")
        for start in range(0, max(0, len(ordered) - SEQUENCE_LENGTH + 1), STRIDE):
            candidate_windows += 1
            window = ordered[start : start + SEQUENCE_LENGTH]
            times = [item["timestamp"] for item in window]
            valid = all(
                (times[i + 1] - times[i]).total_seconds() / 60 == CADENCE_MINUTES
                for i in range(SEQUENCE_LENGTH - 1)
            )
            if not valid:
                cadence_failures += 1
                continue
            endpoint = window[-1]
            inherited = endpoint["flare_label"]
            target = int(str(inherited)[:1] in {"M", "X"})
            item = {
                "sample_id": endpoint["basename"],
                "split": split,
                "active_region": str(ar),
                "endpoint_filename": endpoint["filename"],
                "endpoint_basename": endpoint["basename"],
                "endpoint_timestamp": endpoint["timestamp"].strftime("%Y-%m-%dT%H:%M:%S"),
                "inherited_flare_label": inherited,
                "target_ge_m": target,
            }
            for index, frame in enumerate(window):
                item[f"frame_{index}"] = frame["filename"]
                item[f"frame_{index}_exists"] = (IMAGE_ROOT / frame["filename"]).is_file()
            item["all_frames_exist"] = all(item[f"frame_{index}_exists"] for index in range(SEQUENCE_LENGTH))
            rows.append(item)

    frame = pd.DataFrame(rows)
    summary = {
        "split": split,
        "raw_rows": int(len(pd.read_csv(ROOT / "data" / f"{split}_Data_by_AR_png_224.csv"))),
        "valid_labeled_rows": int(len(source)),
        "raw_active_regions": int(source.ar.nunique()),
        "sequence_count": int(len(frame)),
        "sequence_active_regions": int(frame.active_region.nunique()),
        "candidate_windows": candidate_windows,
        "cadence_failures": cadence_failures,
        "positive_ge_m": int(frame.target_ge_m.sum()),
        "negative_ge_m": int((frame.target_ge_m == 0).sum()),
        "missing_image_sequences": int((~frame.all_frames_exist).sum()) if len(frame) else 0,
    }
    return frame, summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    labels = load_labels()
    frames = []
    summaries = []
    for split in ("Train", "Validation", "Test"):
        frame, summary = make_split(split, labels)
        frame.to_csv(OUT / f"{split.lower()}_all_label_sequences.csv", index=False)
        frames.append(frame)
        summaries.append(summary)

    population = pd.concat(frames, ignore_index=True)
    features = pd.read_csv(ROOT / "data" / "Lat60_Lon60_Nans0_C1.0_24hr_png_224_features.csv", header=None, low_memory=False)
    feature_names = [f"feature_{i:02d}" for i in range(29)]
    feature = features.iloc[:, :29].copy()
    feature.columns = feature_names
    feature["endpoint_basename"] = features.iloc[:, -1].astype("string").str.rsplit("/", n=1).str[-1]
    feature = feature.drop_duplicates("endpoint_basename")
    matched = population.merge(feature, on="endpoint_basename", how="left", indicator=True)
    if not (matched["_merge"] == "both").all():
        raise RuntimeError("Endpoint-to-feature matching is incomplete")
    matched.drop(columns=["_merge"], inplace=True)
    matched.to_csv(OUT / "matched_all_label_endpoint_features.csv", index=False)

    split_sets = {split: set(population.loc[population.split == split, "active_region"]) for split in ("Train", "Validation", "Test")}
    overlaps = {f"{left}_{right}": sorted(split_sets[left] & split_sets[right]) for left in split_sets for right in split_sets if left < right}
    if any(overlaps.values()):
        raise RuntimeError(f"Active-region overlap detected: {overlaps}")
    if population.sample_id.duplicated().any():
        raise RuntimeError("Duplicate endpoint sample IDs detected")

    summary = {
        "splits": summaries,
        "total_sequences": int(len(population)),
        "total_positive_ge_m": int(population.target_ge_m.sum()),
        "total_negative_ge_m": int((population.target_ge_m == 0).sum()),
        "feature_match_rows": int(len(matched)),
        "active_region_overlaps": overlaps,
        "feature_names": feature_names,
    }
    (OUT / "matched_population_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

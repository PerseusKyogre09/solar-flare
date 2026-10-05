from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

FLARE_RE = re.compile(r"^(0|[CMX]\d+(?:\.\d+)?)$")
TARGETS = {"ge_c": {"C", "M", "X"}, "ge_m": {"M", "X"}}


class DataValidationError(ValueError):
    """Raised when the research data cannot be validated safely."""


@dataclass
class DatasetBundle:
    frame: pd.DataFrame
    feature_names: list[str]
    report: dict


def target_from_flare(flare: pd.Series, target: str) -> pd.Series:
    if target not in TARGETS:
        raise ValueError(f"Unknown target {target!r}; choose ge_c or ge_m")
    normalized = flare.astype("string").str.strip().str.upper()
    prefix = normalized.str[:1]
    return prefix.isin(TARGETS[target]).astype("int8")


def parse_filename(path_or_name: pd.Series) -> tuple[pd.Series, pd.Series]:
    names = path_or_name.astype("string").str.replace("\\", "/", regex=False).str.rsplit("/", n=1).str[-1]
    ar = names.str.extract(r"^(\d+)_hmi\.M_", expand=False)
    ts_text = names.str.extract(r"\.(\d{8}_\d{6})_TAI", expand=False)
    timestamp = pd.to_datetime(ts_text, format="%Y%m%d_%H%M%S", errors="coerce")
    return ar, timestamp


def _read_labels(path: Path) -> pd.DataFrame:
    labels = pd.read_csv(path, names=["filename", "flare"], header=None, dtype="string", keep_default_na=False)
    labels["filename"] = labels.filename.str.strip()
    labels["flare"] = labels.flare.str.strip().str.upper()
    valid = labels.flare.map(lambda x: bool(FLARE_RE.fullmatch(str(x))))
    if not valid.all():
        bad = labels.loc[~valid, "flare"].value_counts().head(10).to_dict()
        raise DataValidationError(f"Malformed flare labels found: {bad}")
    if labels.filename.duplicated().any():
        raise DataValidationError("Duplicate image filenames found in label file")
    return labels


def _read_feature_file(path: Path, expected_features: int = 29) -> tuple[pd.DataFrame, list[str]]:
    # The Dryad legacy feature file has no header: 29 numeric features, a
    # derived C1 binary label, the original flare string, and basename.
    raw = pd.read_csv(path, header=None, dtype="string", keep_default_na=False, low_memory=False)
    if len(raw) == 0 or raw.shape[1] < expected_features + 1:
        raise DataValidationError(f"Feature file must contain at least {expected_features + 1} columns")
    first_numeric = pd.to_numeric(raw.iloc[0, :expected_features], errors="coerce").notna().all()
    if first_numeric:
        feature_names = [f"feature_{i:02d}" for i in range(expected_features)]
        frame = raw.iloc[:, :expected_features].copy()
        frame.columns = feature_names
        # Filename is the last column in the documented Dryad output. The
        # preceding columns are labels and are deliberately ignored.
        frame["feature_filename"] = raw.iloc[:, -1]
    else:
        header = pd.read_csv(path, nrows=0).columns.tolist()
        if len(header) != raw.shape[1]:
            raise DataValidationError("Could not consistently parse feature CSV header")
        frame = pd.read_csv(path, low_memory=False)
        lowered = {str(c).lower(): c for c in frame.columns}
        filename_col = next((c for k, c in lowered.items() if any(x in k for x in ("filename", "image", "path"))), None)
        if filename_col is None:
            raise DataValidationError("Headered feature CSV has no identifiable filename column")
        excluded = {filename_col}
        excluded |= {c for k, c in lowered.items() if any(x in k for x in ("label", "flare", "class", "target", "ar", "region", "time", "date"))}
        candidates = [c for c in frame.columns if c not in excluded and pd.api.types.is_numeric_dtype(frame[c])]
        if len(candidates) != expected_features:
            raise DataValidationError(f"Expected exactly {expected_features} numeric feature columns; found {len(candidates)}: {candidates}")
        feature_names = [str(c) for c in candidates]
        frame = frame[feature_names + [filename_col]].rename(columns={filename_col: "feature_filename"})
    for col in feature_names:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if frame.feature_filename.isna().any() or (frame.feature_filename.astype("string").str.strip() == "").any():
        raise DataValidationError("Feature file contains missing filenames")
    if frame.feature_filename.duplicated().any():
        raise DataValidationError("Duplicate filenames found in feature file")
    return frame, feature_names


def _split_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype="string")
    if "filename" not in df.columns:
        raise DataValidationError(f"Split file {path} has no filename column")
    df["active_region"], df["timestamp"] = parse_filename(df.filename)
    if df.active_region.isna().any() or df.timestamp.isna().any():
        raise DataValidationError(f"Malformed active-region ID or timestamp in {path}")
    return df[["filename", "active_region", "timestamp"]]


def split_overlap(split_frames: dict[str, pd.DataFrame]) -> dict[str, list[str]]:
    names = list(split_frames)
    overlap = {}
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            common = sorted(set(split_frames[left].active_region) & set(split_frames[right].active_region))
            overlap[f"{left}_{right}"] = common
    return overlap


def load_dataset(feature_path: str | Path, label_path: str | Path, split_dir: str | Path, target: str) -> DatasetBundle:
    feature_path, label_path, split_dir = map(Path, (feature_path, label_path, split_dir))
    for path in (feature_path, label_path, split_dir):
        if not path.exists():
            raise FileNotFoundError(f"Required dataset path does not exist: {path}")
    labels = _read_labels(label_path)
    features, feature_names = _read_feature_file(feature_path)
    features["feature_basename"] = features.feature_filename.astype("string").str.replace("\\", "/", regex=False).str.rsplit("/", n=1).str[-1]
    if features.feature_basename.duplicated().any():
        raise DataValidationError("Feature filenames are not unique after basename normalization")
    label_map = labels.set_index("filename").flare
    features["flare"] = features.feature_basename.map(label_map)
    if features.flare.isna().any():
        raise DataValidationError(f"{int(features.flare.isna().sum())} feature rows do not match the flare label file")
    features["active_region"], features["timestamp"] = parse_filename(features.feature_basename)
    if features.active_region.isna().any() or features.timestamp.isna().any():
        raise DataValidationError("Feature file contains malformed active-region IDs or timestamps")
    split_frames = {name: _split_frame(split_dir / f"{name}_Data_by_AR_png_224.csv") for name in ("Train", "Validation", "Test")}
    overlap = split_overlap(split_frames)
    if any(overlap.values()):
        raise DataValidationError(f"Active-region leakage between predefined splits: {overlap}")
    split_map = pd.concat([df.assign(split=name) for name, df in split_frames.items()], ignore_index=True)
    if split_map.filename.duplicated().any():
        raise DataValidationError("Duplicate image filenames found across split files")
    split_map["basename"] = split_map.filename.str.rsplit("/", n=1).str[-1]
    merged = features.merge(split_map[["basename", "split", "active_region", "timestamp"]], left_on="feature_basename", right_on="basename", how="inner", validate="one_to_one", suffixes=("_feature", "_split"))
    if len(merged) != sum(len(x) for x in split_frames.values()):
        raise DataValidationError("Feature/split alignment is incomplete; refusing to train on a partial join")
    if not (merged.active_region_feature == merged.active_region_split).all() or not (merged.timestamp_feature == merged.timestamp_split).all():
        raise DataValidationError("Feature filenames do not exactly match the split filename active-region/timestamp")
    merged["active_region"] = merged.pop("active_region_feature")
    merged["timestamp"] = merged.pop("timestamp_feature")
    merged.drop(columns=["active_region_split", "timestamp_split"], inplace=True)
    merged["target"] = target_from_flare(merged.flare, target)
    merged["row_id"] = merged.feature_basename
    missing = merged[feature_names].isna().sum()
    infinite = pd.Series({c: int(np.isinf(merged[c].to_numpy(dtype=float)).sum()) for c in feature_names})
    merged[feature_names] = merged[feature_names].replace([np.inf, -np.inf], np.nan)
    report = {
        "feature_file": str(feature_path), "label_file": str(label_path), "target": target,
        "total_observations": int(len(merged)), "usable_observations": int(len(merged)),
        "feature_count": len(feature_names), "feature_names": feature_names,
        "missing_value_counts": {str(k): int(v) for k, v in missing.items()},
        "infinite_value_counts": {str(k): int(v) for k, v in infinite.items()},
        "duplicate_rows": int(merged.duplicated(subset=feature_names + ["feature_basename", "flare"]).sum()),
        "active_regions": int(merged.active_region.nunique()),
        "class_distribution": {str(k): int(v) for k, v in merged.target.value_counts().sort_index().items()},
        "flare_distribution": {str(k): int(v) for k, v in merged.flare.value_counts().items()},
        "timestamp_min": str(merged.timestamp.min()), "timestamp_max": str(merged.timestamp.max()),
        "observations_per_active_region": {str(k): int(v) for k, v in merged.groupby("active_region", sort=True).size().describe().to_dict().items()},
        "positive_active_regions": int(merged.loc[merged.target.eq(1), "active_region"].nunique()),
        "split_active_regions": {name: int(df.active_region.nunique()) for name, df in split_frames.items()},
        "split_timestamp_ranges": {name: [str(df.timestamp.min()), str(df.timestamp.max())] for name, df in split_frames.items()},
        "split_overlap": {k: v for k, v in overlap.items()},
        "label_semantics_verified": "0 means no qualifying flare; C/M/X are classed by letter prefix in the checked-in label file",
    }
    return DatasetBundle(merged, feature_names, report)

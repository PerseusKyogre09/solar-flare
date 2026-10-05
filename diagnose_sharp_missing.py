import re
from pathlib import Path

import pandas as pd


SEQUENCE_FILES = [
    "data/Train_sequences.csv",
    "data/Validation_sequences.csv",
    "data/Test_sequences.csv",
]

CACHE_FILE = "data/sharp_cache/sharp_12min_cache.csv"


def parse_filename(filename):
    m = re.search(
        r"^(\d+)/.*?(\d{8})_(\d{6})_TAI",
        str(filename),
    )

    if not m:
        return None

    ar = int(m.group(1))

    timestamp = pd.to_datetime(
        m.group(2) + m.group(3),
        format="%Y%m%d%H%M%S",
    )

    return ar, timestamp


# ------------------------------------------------------------
# Load requested frames
# ------------------------------------------------------------

requested = []

for file in SEQUENCE_FILES:

    print("Reading:", file)

    df = pd.read_csv(file)

    frame_cols = [
        c for c in df.columns
        if c.startswith("frame_")
    ]

    for col in frame_cols:

        for filename in df[col].dropna():

            parsed = parse_filename(filename)

            if parsed:
                ar, timestamp = parsed

                requested.append(
                    (ar, timestamp)
                )


requested_df = pd.DataFrame(
    requested,
    columns=["dataset_ar", "timestamp"],
)

requested_df = requested_df.drop_duplicates()

print()
print("Unique requested frames:", len(requested_df))


# ------------------------------------------------------------
# Load SHARP cache
# ------------------------------------------------------------

cache = pd.read_csv(
    CACHE_FILE,
    parse_dates=["timestamp"],
)

cache_keys = cache[
    ["dataset_ar", "timestamp"]
].drop_duplicates()

print(
    "Cached frames:",
    len(cache_keys),
)


# ------------------------------------------------------------
# Find missing frames
# ------------------------------------------------------------

merged = requested_df.merge(
    cache_keys,
    on=["dataset_ar", "timestamp"],
    how="left",
    indicator=True,
)

missing = merged[
    merged["_merge"] == "left_only"
].drop(columns="_merge")


print()
print("=" * 70)
print("MISSING FRAME ANALYSIS")
print("=" * 70)

print(
    "Missing unique frames:",
    len(missing),
)


# ------------------------------------------------------------
# Missing frames by AR
# ------------------------------------------------------------

print()
print("Top ARs with missing frames:")

missing_by_ar = (
    missing
    .groupby("dataset_ar")
    .size()
    .sort_values(ascending=False)
)

print(
    missing_by_ar.head(30).to_string()
)


# ------------------------------------------------------------
# Requested / matched / missing by AR
# ------------------------------------------------------------

requested_by_ar = (
    requested_df
    .groupby("dataset_ar")
    .size()
)

matched_by_ar = (
    cache_keys
    .groupby("dataset_ar")
    .size()
)

summary = pd.DataFrame({
    "requested": requested_by_ar,
    "matched": matched_by_ar,
}).fillna(0)

summary["missing"] = (
    summary["requested"]
    - summary["matched"]
)

summary["match_rate"] = (
    summary["matched"]
    / summary["requested"]
    * 100
)

summary = summary.sort_values(
    "missing",
    ascending=False,
)

print()
print("=" * 70)
print("AR MATCH SUMMARY")
print("=" * 70)

print(
    summary.head(50).to_string()
)


# ------------------------------------------------------------
# Save missing frames
# ------------------------------------------------------------

missing.to_csv(
    "data/sharp_cache/missing_frames.csv",
    index=False,
)

summary.to_csv(
    "data/sharp_cache/sharp_match_summary.csv"
)

print()
print("Saved:")
print("data/sharp_cache/missing_frames.csv")
print("data/sharp_cache/sharp_match_summary.csv")


# ------------------------------------------------------------
# Time gaps
# ------------------------------------------------------------

missing_sorted = missing.sort_values(
    ["dataset_ar", "timestamp"]
)

print()
print("=" * 70)
print("SAMPLE MISSING FRAMES")
print("=" * 70)

print(
    missing_sorted.head(50).to_string(
        index=False
    )
)

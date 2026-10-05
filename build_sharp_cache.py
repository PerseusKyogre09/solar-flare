import re
import time
from pathlib import Path

import drms
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

SEQUENCE_FILES = [
    "data/Train_sequences.csv",
    "data/Validation_sequences.csv",
    "data/Test_sequences.csv",
]

CACHE_DIR = Path("data/sharp_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = CACHE_DIR / "sharp_12min_cache.csv"

# Dataset AR -> NOAA AR
# 1069 -> 11069
# 1087 -> 11087
# 1089 -> 11089
NOAA_OFFSET = 10000

# Physical SHARP features
SHARP_FEATURES = [
    "USFLUX",
    "MEANGAM",
    "MEANGBT",
    "MEANGBZ",
    "MEANGBH",
    "MEANJZD",
    "TOTUSJZ",
    "MEANALP",
    "MEANJZH",
    "TOTUSJH",
    "ABSNJZH",
    "SAVNCPP",
    "MEANPOT",
    "TOTPOT",
    "MEANSHR",
    "SHRGT45",
    "R_VALUE",
    "AREA_ACR",
]

SHARP_KEYS = [
    "HARPNUM",
    "T_REC",
    "NOAA_AR",
    *SHARP_FEATURES,
]


# ============================================================
# PARSE IMAGE FILENAME
# ============================================================

def parse_frame_filename(filename):
    """
    Example:

    1069/1069_hmi.M_720s.20100505_000000_TAI.1.magnetogram_224.png

    Returns:

        dataset_ar
        timestamp
    """

    match = re.search(
        r"^(\d+)/.*?(\d{8})_(\d{6})_TAI",
        str(filename),
    )

    if not match:
        raise ValueError(
            f"Could not parse frame filename:\n{filename}"
        )

    dataset_ar = int(match.group(1))

    timestamp = pd.to_datetime(
        match.group(2) + match.group(3),
        format="%Y%m%d%H%M%S",
    )

    return dataset_ar, timestamp


# ============================================================
# LOAD UNIQUE FRAMES
# ============================================================

print("=" * 70)
print("BUILDING SHARP PHYSICAL DATA CACHE")
print("=" * 70)

all_frames = []

for sequence_file in SEQUENCE_FILES:

    print(f"\nReading: {sequence_file}")

    df = pd.read_csv(sequence_file)

    frame_columns = [
        c for c in df.columns
        if c.startswith("frame_")
    ]

    print(f"Sequences: {len(df):,}")
    print(f"Frame columns: {len(frame_columns)}")

    for column in frame_columns:

        for filename in df[column].dropna():

            ar, timestamp = parse_frame_filename(
                filename
            )

            all_frames.append(
                (ar, timestamp)
            )


unique_frames = sorted(
    set(all_frames)
)

print("\n" + "=" * 70)
print("FRAME SUMMARY")
print("=" * 70)

print(
    f"Total frame references: "
    f"{len(all_frames):,}"
)

print(
    f"Unique AR/timestamp pairs: "
    f"{len(unique_frames):,}"
)

unique_ars = sorted(
    set(ar for ar, _ in unique_frames)
)

print(
    f"Unique active regions: "
    f"{len(unique_ars):,}"
)


# ============================================================
# JSOC CONNECTION
# ============================================================

print("\nConnecting to JSOC...")

client = drms.Client()

print("Connected.")


# ============================================================
# STEP 1
# RESOLVE HARP FOR EACH AR
#
# We do NOT use the static HARP mapping file.
#
# Instead, we query JSOC around a representative timestamp
# and ask specifically for the NOAA AR.
# ============================================================

print("\n" + "=" * 70)
print("RESOLVING HARPs")
print("=" * 70)

# We only need to resolve each AR once.
#
# Use the earliest timestamp belonging to each AR as the
# representative timestamp.

ar_timestamps = {}

for ar, timestamp in unique_frames:

    if ar not in ar_timestamps:
        ar_timestamps[ar] = timestamp


arp_to_harp = {}

missing_harps = []

for index, ar in enumerate(
    sorted(ar_timestamps),
    start=1,
):

    timestamp = ar_timestamps[ar]

    noaa_ar = NOAA_OFFSET + ar

    start = (
        timestamp - pd.Timedelta(minutes=12)
    ).strftime(
        "%Y.%m.%d_%H:%M:%S"
    )

    end = (
        timestamp + pd.Timedelta(minutes=12)
    ).strftime(
        "%Y.%m.%d_%H:%M:%S"
    )

    query = (
        f"hmi.sharp_720s[]"
        f"[{start}_TAI-{end}_TAI]"
        f"[? NOAA_AR = {noaa_ar} ?]"
    )

    try:

        result = client.query(
            query,
            key=[
                "HARPNUM",
                "T_REC",
                "NOAA_AR",
            ],
        )

    except Exception as e:

        print(
            f"[{index}/{len(ar_timestamps)}] "
            f"AR {ar} query failed: {e}"
        )

        missing_harps.append(ar)

        continue

    if len(result) == 0:

        print(
            f"[{index}/{len(ar_timestamps)}] "
            f"AR {ar} → NOAA {noaa_ar} → NO MATCH"
        )

        missing_harps.append(ar)

        continue

    result["timestamp"] = pd.to_datetime(
        result["T_REC"]
        .astype(str)
        .str.replace(
            "_TAI",
            "",
            regex=False,
        ),
        format="%Y.%m.%d_%H:%M:%S",
    )

    # Only accept records that actually correspond to
    # the representative timestamp.

    exact = result[
        result["timestamp"] == timestamp
    ]

    if len(exact) == 0:

        print(
            f"[{index}/{len(ar_timestamps)}] "
            f"AR {ar} → NOAA {noaa_ar} → "
            f"no exact timestamp"
        )

        missing_harps.append(ar)

        continue

    # Usually one HARP.
    # If multiple exist, select the exact matching record.
    harp = int(
        exact.iloc[0]["HARPNUM"]
    )

    arp_to_harp[ar] = harp

    print(
        f"[{index}/{len(ar_timestamps)}] "
        f"AR {ar} → NOAA {noaa_ar} → "
        f"HARP {harp}"
    )

    time.sleep(0.05)


print("\nHARP resolution complete.")

print(
    f"Resolved: "
    f"{len(arp_to_harp):,}"
)

print(
    f"Missing: "
    f"{len(missing_harps):,}"
)


# ============================================================
# STEP 2
# GROUP ALL REQUESTED FRAMES BY HARP
# ============================================================

print("\n" + "=" * 70)
print("GROUPING FRAMES BY HARP")
print("=" * 70)

harp_requests = {}

for ar, timestamp in unique_frames:

    if ar not in arp_to_harp:
        continue

    harp = arp_to_harp[ar]

    noaa_ar = NOAA_OFFSET + ar

    harp_requests.setdefault(
        harp,
        []
    ).append(
        {
            "dataset_ar": ar,
            "noaa_ar": noaa_ar,
            "timestamp": timestamp,
        }
    )


print(
    f"Unique HARPs: "
    f"{len(harp_requests):,}"
)


# ============================================================
# STEP 3
# DOWNLOAD SHARP RECORDS BY HARP
# ============================================================

print("\n" + "=" * 70)
print("DOWNLOADING SHARP RECORDS")
print("=" * 70)

cache_rows = []

matched = 0
missing_records = 0

for index, (harp, requests) in enumerate(
    harp_requests.items(),
    start=1,
):

    timestamps = [
        item["timestamp"]
        for item in requests
    ]

    start_time = min(timestamps)
    end_time = max(timestamps)

    print(
        f"\n[{index}/{len(harp_requests)}] "
        f"HARP {harp}"
    )

    print(
        f"Requested frames: "
        f"{len(requests):,}"
    )

    print(
        f"Range: "
        f"{start_time} → {end_time}"
    )

    start_jsoc = (
        start_time.strftime(
            "%Y.%m.%d_%H:%M:%S"
        )
        + "_TAI"
    )

    end_jsoc = (
        end_time.strftime(
            "%Y.%m.%d_%H:%M:%S"
        )
        + "_TAI"
    )

    query = (
        f"hmi.sharp_720s[{harp}]"
        f"[{start_jsoc}-{end_jsoc}]"
    )

    try:

        result = client.query(
            query,
            key=SHARP_KEYS,
        )

    except Exception as e:

        print(
            f"ERROR querying HARP {harp}:"
        )

        print(e)

        missing_records += len(requests)

        continue

    print(
        f"SHARP records returned: "
        f"{len(result):,}"
    )

    if len(result) == 0:

        missing_records += len(requests)

        continue

    # Normalize JSOC timestamps
    result["timestamp"] = pd.to_datetime(
        result["T_REC"]
        .astype(str)
        .str.replace(
            "_TAI",
            "",
            regex=False,
        ),
        format="%Y.%m.%d_%H:%M:%S",
    )

    # --------------------------------------------------------
    # Create timestamp lookup
    # --------------------------------------------------------

    result_lookup = {}

    for _, row in result.iterrows():

        result_lookup[
            row["timestamp"]
        ] = row

    # --------------------------------------------------------
    # Match requested timestamps
    # --------------------------------------------------------

    for request in requests:

        timestamp = request["timestamp"]

        if timestamp not in result_lookup:

            missing_records += 1

            continue

        row = result_lookup[timestamp]

        # Extra safety:
        #
        # Verify NOAA AR before accepting the record.

        expected_noaa = request["noaa_ar"]

        actual_noaa = int(
            row["NOAA_AR"]
        )

        if actual_noaa != expected_noaa:

            print(
                "WARNING: NOAA AR mismatch!"
            )

            print(
                f"Expected: {expected_noaa}"
            )

            print(
                f"Actual:   {actual_noaa}"
            )

            missing_records += 1

            continue

        output = {
            "dataset_ar": request["dataset_ar"],
            "noaa_ar": expected_noaa,
            "harpnum": harp,
            "timestamp": timestamp,
        }

        for feature in SHARP_FEATURES:

            output[feature] = row[feature]

        cache_rows.append(output)

        matched += 1

    time.sleep(0.1)


# ============================================================
# SAVE CACHE
# ============================================================

print("\n" + "=" * 70)
print("SAVING SHARP CACHE")
print("=" * 70)

if cache_rows:

    cache_df = pd.DataFrame(
        cache_rows
    )

    cache_df = cache_df.sort_values(
        [
            "dataset_ar",
            "timestamp",
        ]
    )

    cache_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

else:

    cache_df = pd.DataFrame()

    print(
        "WARNING: No records were matched."
    )


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("SHARP EXTRACTION COMPLETE")
print("=" * 70)

total = len(unique_frames)

print(
    f"Unique requested frames: "
    f"{total:,}"
)

print(
    f"ARs resolved to HARP: "
    f"{len(arp_to_harp):,}"
)

print(
    f"ARs without HARP: "
    f"{len(missing_harps):,}"
)

print(
    f"Matched SHARP frames: "
    f"{matched:,}"
)

print(
    f"Missing SHARP frames: "
    f"{missing_records:,}"
)

if total:

    print(
        f"\nOverall match rate: "
        f"{matched / total * 100:.4f}%"
    )

print(
    f"\nSaved:"
)

print(
    OUTPUT_FILE
)

if len(cache_df):

    print("\nCache shape:")

    print(
        cache_df.shape
    )

    print("\nCache preview:")

    print(
        cache_df.head(5).to_string(
            index=False
        )
    )

print("\nDone.")
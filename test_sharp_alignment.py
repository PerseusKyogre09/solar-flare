import re
import pandas as pd
import drms

# ============================================================
# Configuration
# ============================================================

SEQUENCE_FILE = "data/Train_sequences.csv"
MAX_SEQUENCES = 100

client = drms.Client()

# Physical SHARP features we eventually want
SHARP_KEYS = [
    "HARPNUM",
    "T_REC",
    "NOAA_AR",
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


# ============================================================
# Helpers
# ============================================================

def parse_frame(filename):
    """
    Example:
    1069/1069_hmi.M_720s.20100505_000000_TAI.1.magnetogram_224.png

    Returns:
        dataset_ar = 1069
        timestamp  = 2010-05-05 00:00:00
    """

    m = re.match(
        r"(\d+)/.*?(\d{8})_(\d{6})_TAI",
        filename
    )

    if not m:
        raise ValueError(f"Could not parse: {filename}")

    ar = int(m.group(1))

    timestamp = pd.to_datetime(
        m.group(2) + m.group(3),
        format="%Y%m%d%H%M%S"
    )

    return ar, timestamp


def dataset_ar_to_noaa(ar):
    """
    The dataset uses 1069 while SHARP metadata uses 11069.
    """

    return 11000 + ar


# ============================================================
# Load sequences
# ============================================================

df = pd.read_csv(SEQUENCE_FILE)

df = df.head(MAX_SEQUENCES)

print("=" * 60)
print("SHARP ALIGNMENT TEST")
print("=" * 60)

print(f"Testing sequences: {len(df)}")


# ============================================================
# Collect unique frames
# ============================================================

frames = {}

for _, row in df.iterrows():

    for i in range(10):

        filename = row[f"frame_{i}"]

        ar, timestamp = parse_frame(filename)

        key = (ar, timestamp)

        frames[key] = {
            "dataset_ar": ar,
            "timestamp": timestamp,
        }


print(f"Unique frames: {len(frames)}")


# ============================================================
# Test first few mappings
# ============================================================

results = []

for n, ((ar, timestamp), info) in enumerate(frames.items()):

    noaa_ar = dataset_ar_to_noaa(ar)

    # First find the HARP associated with this NOAA AR.
    #
    # We use JSOC metadata here rather than assuming
    # NOAA AR == HARPNUM.

    mapping_query = (
        f'hmi.sharp_720s[]['
        f'{timestamp.strftime("%Y.%m.%d_%H:%M:%S")}_TAI'
        f'][? NOAA_ARS ~ "{noaa_ar}" ?]'
    )

    try:

        mapping = client.query(
            mapping_query,
            key=["HARPNUM", "T_REC", "NOAA_AR"]
        )

    except Exception as e:

        print(f"\nJSOC query failed for AR {ar}: {e}")
        continue

    if len(mapping) == 0:

        print(
            f"NO MATCH | AR {ar} | "
            f"NOAA {noaa_ar} | {timestamp}"
        )

        results.append({
            **info,
            "noaa_ar": noaa_ar,
            "harpnum": None,
            "matched": False,
        })

        continue

    harpnums = mapping["HARPNUM"].unique()

    # Usually there should be one relevant HARP.
    # If multiple are returned, we'll inspect them later.

    harp = int(harpnums[0])

    # Query exact physical record
    physical_query = (
        f'hmi.sharp_720s[{harp}]'
        f'[{timestamp.strftime("%Y.%m.%d_%H:%M:%S")}_TAI]'
    )

    physical = client.query(
        physical_query,
        key=SHARP_KEYS
    )

    if len(physical) == 0:

        print(
            f"NO PHYSICAL DATA | "
            f"AR {ar} | NOAA {noaa_ar} | "
            f"HARP {harp} | {timestamp}"
        )

        results.append({
            **info,
            "noaa_ar": noaa_ar,
            "harpnum": harp,
            "matched": False,
        })

        continue

    row_physical = physical.iloc[0]

    print(
        f"OK | AR {ar} | "
        f"NOAA {noaa_ar} | "
        f"HARP {harp} | "
        f"{timestamp}"
    )

    results.append({
        **info,
        "noaa_ar": noaa_ar,
        "harpnum": harp,
        "matched": True,
        "USFLUX": row_physical["USFLUX"],
        "TOTUSJH": row_physical["TOTUSJH"],
        "TOTUSJZ": row_physical["TOTUSJZ"],
        "TOTPOT": row_physical["TOTPOT"],
        "MEANSHR": row_physical["MEANSHR"],
        "R_VALUE": row_physical["R_VALUE"],
        "AREA_ACR": row_physical["AREA_ACR"],
    })

    # Don't hammer JSOC during the test
    if n >= 99:
        break


# ============================================================
# Summary
# ============================================================

results_df = pd.DataFrame(results)

matched = results_df["matched"].sum()
total = len(results_df)

print("\n" + "=" * 60)
print("RESULT")
print("=" * 60)

print(f"Frames tested: {total}")
print(f"Matched:       {matched}")
print(f"Missing:       {total - matched}")

if total:
    print(f"Match rate:    {matched / total * 100:.2f}%")

results_df.to_csv(
    "data/sharp_alignment_test.csv",
    index=False
)

print("\nSaved:")
print("data/sharp_alignment_test.csv")

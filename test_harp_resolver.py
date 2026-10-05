import drms
import pandas as pd


client = drms.Client()

tests = [
    (1069, "2010-05-05 00:00:00"),
    (1087, "2010-07-17 17:00:00"),
    (1089, "2010-07-26 04:36:00"),
]


def resolve_harp(dataset_ar, timestamp):

    noaa_ar = 10000 + dataset_ar

    ts = pd.Timestamp(timestamp)

    # Ask JSOC for SHARP records around this exact timestamp.
    # We deliberately search ALL HARPs, then identify the
    # record whose NOAA_AR and timestamp actually match.

    start = (ts - pd.Timedelta(minutes=12)).strftime(
        "%Y.%m.%d_%H:%M:%S"
    )
    end = (ts + pd.Timedelta(minutes=12)).strftime(
        "%Y.%m.%d_%H:%M:%S"
    )

    query = (
        f"hmi.sharp_720s[]"
        f"[{start}_TAI-{end}_TAI]"
        f"[? NOAA_AR = {noaa_ar} ?]"
    )

    print("\nQUERY:")
    print(query)

    result = client.query(
        query,
        key=[
            "HARPNUM",
            "T_REC",
            "NOAA_AR",
        ],
    )

    if len(result) == 0:
        return None, result

    result["timestamp"] = pd.to_datetime(
        result["T_REC"]
        .astype(str)
        .str.replace("_TAI", "", regex=False),
        format="%Y.%m.%d_%H:%M:%S",
    )

    exact = result[
        result["timestamp"] == ts
    ]

    if len(exact) == 0:
        return None, result

    return int(exact.iloc[0]["HARPNUM"]), exact


print("=" * 70)
print("TIMESTAMP-AWARE HARP RESOLVER TEST")
print("=" * 70)

for dataset_ar, timestamp in tests:

    print("\n" + "=" * 70)

    print(
        f"Dataset AR: {dataset_ar}"
    )

    print(
        f"Timestamp: {timestamp}"
    )

    harp, result = resolve_harp(
        dataset_ar,
        timestamp,
    )

    if harp is None:

        print("NO MATCH")

        if len(result):
            print("\nNearby records:")
            print(
                result.to_string(
                    index=False
                )
            )

    else:

        print(
            f"FOUND HARP: {harp}"
        )

        print("\nMatched record:")
        print(
            result.to_string(
                index=False
            )
        )

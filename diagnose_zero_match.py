import pandas as pd
import drms

MISSING = "data/sharp_cache/missing_frames.csv"

df = pd.read_csv(MISSING, parse_dates=["timestamp"])

# Largest zero-match ARs from your previous output
ars = [
    2056,
    1968,
    1520,
    2080,
    2157,
    2149,
    2146,
    1499,
    1865,
    2324,
]

client = drms.Client()

print("=" * 70)
print("ZERO-MATCH AR DIAGNOSTIC")
print("=" * 70)

for ar in ars:

    rows = df[df["dataset_ar"] == ar]

    if rows.empty:
        continue

    timestamp = rows["timestamp"].iloc[0]

    # Dataset AR -> NOAA AR convention
    noaa = 10000 + ar

    start = timestamp - pd.Timedelta(minutes=12)
    end   = timestamp + pd.Timedelta(minutes=12)

    start_str = start.strftime("%Y.%m.%d_%H:%M:%S_TAI")
    end_str   = end.strftime("%Y.%m.%d_%H:%M:%S_TAI")

    query = (
        f'hmi.sharp_720s[]'
        f'[{start_str}-{end_str}]'
        f'[? NOAA_AR = {noaa} ?]'
    )

    print()
    print("=" * 70)
    print(f"DATASET AR: {ar}")
    print(f"NOAA AR:    {noaa}")
    print(f"TIMESTAMP:  {timestamp}")
    print()
    print("QUERY:")
    print(query)

    try:
        result = client.query(
            query,
            key=[
                "HARPNUM",
                "T_REC",
                "NOAA_AR",
            ]
        )

        if len(result) == 0:
            print("RESULT: NO MATCH")
        else:
            print(f"RESULT: {len(result)} records")
            print(result.to_string(index=False))

    except Exception as e:
        print("ERROR:", e)

import pandas as pd
from pathlib import Path

LABEL_FILE = Path("data/C1.0_24hr_224_png_Labels.txt")

SEQUENCE_LENGTH = 10
STRIDE = 10
MAX_GAP_MINUTES = 12


# =========================
# Load labels
# =========================

labels = {}

with open(LABEL_FILE) as f:
    for line in f:
        filename, flare = line.strip().split(",", 1)
        labels[filename] = flare


def get_class(flare):
    if flare.startswith("C"):
        return "C"
    if flare.startswith("M"):
        return "M"
    if flare.startswith("X"):
        return "X"
    return None


# =========================
# Process split
# =========================

def make_sequences(split):

    csv_file = Path(f"data/{split}_Data_by_AR_png_224.csv")

    df = pd.read_csv(csv_file)

    df["ar"] = df["filename"].str.split("/", n=1).str[0]
    df["name"] = df["filename"].str.split("/", n=1).str[1]

    df["flare"] = df["name"].map(labels)
    df["class"] = df["flare"].apply(get_class)

    # Extract timestamp
    df["time"] = pd.to_datetime(
        df["name"].str.extract(
            r"\.(\d{8}_\d{6})_TAI"
        )[0],
        format="%Y%m%d_%H%M%S"
    )

    df = df.dropna(subset=["class", "time"])

    sequences = []

    for ar, group in df.groupby("ar"):

        group = group.sort_values("time")

        rows = group.to_dict("records")

        for start in range(
            0,
            len(rows) - SEQUENCE_LENGTH + 1,
            STRIDE
        ):

            window = rows[
                start:start + SEQUENCE_LENGTH
            ]

            # Check temporal continuity
            times = [
                row["time"]
                for row in window
            ]

            valid = True

            for i in range(len(times) - 1):

                gap = (
                    times[i + 1] - times[i]
                ).total_seconds() / 60

                if gap != MAX_GAP_MINUTES:
                    valid = False
                    break

            if not valid:
                continue

            # Target is the final frame
            target = window[-1]["class"]

            sequence = {
                "ar": ar,
                "target": target
            }

            for i, row in enumerate(window):
                sequence[f"frame_{i}"] = row["filename"]

            sequences.append(sequence)

    output = Path(
        f"data/{split}_sequences.csv"
    )

    result = pd.DataFrame(sequences)

    result.to_csv(
        output,
        index=False
    )

    print(
        f"{split}: "
        f"{len(result):,} sequences"
    )

    if len(result):
        print(
            result["target"]
            .value_counts()
            .to_dict()
        )


# =========================
# Generate datasets
# =========================

for split in [
    "Train",
    "Validation",
    "Test"
]:
    make_sequences(split)
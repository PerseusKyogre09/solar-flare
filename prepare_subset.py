import csv
import random
from collections import defaultdict

random.seed(42)

# Load actual flare labels
labels = {}

with open("data/C1.0_24hr_224_png_Labels.txt") as f:
    for line in f:
        filename, flare = line.strip().split(",")
        labels[filename] = flare

# Find training images for C/M/X
by_class = defaultdict(list)

with open("data/Train_Data_by_AR_png_224.csv") as f:
    reader = csv.DictReader(f)

    for row in reader:
        filename = row["filename"]
        basename = filename.split("/", 1)[1]

        flare = labels.get(basename)

        if flare:
            if flare.startswith("C"):
                by_class["C"].append(filename)
            elif flare.startswith("M"):
                by_class["M"].append(filename)
            elif flare.startswith("X"):
                by_class["X"].append(filename)

# Pick 100 from each class
selected = []

for cls in ["C", "M", "X"]:
    random.shuffle(by_class[cls])
    selected.extend(by_class[cls][:100])
    print(f"{cls}: selected 100 / {len(by_class[cls])}")

# Archive has an extra top-level directory
with open("subset_files.txt", "w") as f:
    for filename in selected:
        f.write("Lat60_Lon60_Nans0_png_224/" + filename + "\n")

print(f"\nTotal images selected: {len(selected)}")

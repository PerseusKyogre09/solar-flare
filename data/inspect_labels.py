import csv
from collections import Counter

labels = {}

with open("data/C1.0_24hr_224_png_Labels.txt", "r") as f:
    for line in f:
        filename, flare = line.strip().split(",")
        labels[filename] = flare

counts = Counter()

with open("data/Train_Data_by_AR_png_224.csv", "r") as f:
    reader = csv.DictReader(f)

    for row in reader:
        filename = row["filename"].split("/", 1)[1]
        flare = labels.get(filename)

        if flare:
            if flare.startswith("X"):
                counts["X"] += 1
            elif flare.startswith("M"):
                counts["M"] += 1
            elif flare.startswith("C"):
                counts["C"] += 1
            else:
                counts["Other"] += 1

print(counts)
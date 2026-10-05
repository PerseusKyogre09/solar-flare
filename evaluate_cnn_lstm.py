import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score
)

import matplotlib.pyplot as plt


# ============================================================
# 1. Configuration
# ============================================================

DATA_DIR = Path(
    "data/images/Lat60_Lon60_Nans0_png_224"
)

MODEL_FILE = "cnn_lstm_best.pth"

SEQUENCE_LENGTH = 10
BATCH_SIZE = 8

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASS_NAMES = ["C", "M", "X"]
NUM_CLASSES = 3


print("=" * 60)
print("CNN-LSTM TEST EVALUATION")
print("=" * 60)

print("Model:", MODEL_FILE)
print("Device:", DEVICE)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# 2. Dataset
# ============================================================

class SolarSequenceDataset(Dataset):

    def __init__(self, csv_file):

        self.df = pd.read_csv(csv_file)

        self.class_map = {
            "C": 0,
            "M": 1,
            "X": 2
        }

        self.transform = transforms.Compose([
            transforms.ToTensor(),

            transforms.Normalize(
                (0.5,),
                (0.5,)
            )
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        images = []

        for i in range(SEQUENCE_LENGTH):

            image_path = (
                DATA_DIR /
                row[f"frame_{i}"]
            )

            image = Image.open(
                image_path
            ).convert("L")

            image = self.transform(image)

            images.append(image)

        # [10, 1, 224, 224]
        images = torch.stack(images)

        label = self.class_map[
            row["target"]
        ]

        return images, label


# ============================================================
# 3. Load test dataset
# ============================================================

test_dataset = SolarSequenceDataset(
    "data/Test_sequences.csv"
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

print(
    "\nTest sequences:",
    len(test_dataset)
)


# ============================================================
# 4. CNN Encoder
# ============================================================
#
# IMPORTANT:
# This MUST match the architecture used during training.
#
# Training used:
#
# Conv
# BatchNorm
# ReLU
# MaxPool
#
# Conv
# BatchNorm
# ReLU
# MaxPool
#
# Conv
# BatchNorm
# ReLU
# AdaptiveAvgPool
#
# ============================================================

class CNNEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                1,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(16),

            nn.ReLU(),

            nn.MaxPool2d(2),


            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(2),


            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = x.view(
            x.size(0),
            -1
        )

        return x


# ============================================================
# 5. CNN-LSTM
# ============================================================

class CNNLSTM(nn.Module):

    def __init__(self):

        super().__init__()

        self.cnn = CNNEncoder()

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                64,
                3
            )
        )

    def forward(self, x):

        batch_size = x.size(0)

        sequence_length = x.size(1)


        # ----------------------------------------------------
        # CNN processes each frame
        # ----------------------------------------------------

        x = x.view(
            batch_size * sequence_length,
            1,
            224,
            224
        )

        x = self.cnn(x)


        # ----------------------------------------------------
        # Restore temporal dimension
        # ----------------------------------------------------

        x = x.view(
            batch_size,
            sequence_length,
            64
        )


        # ----------------------------------------------------
        # LSTM
        # ----------------------------------------------------

        x, _ = self.lstm(x)


        # ----------------------------------------------------
        # Final timestep
        # ----------------------------------------------------

        x = x[:, -1, :]


        # ----------------------------------------------------
        # Classifier
        # ----------------------------------------------------

        return self.classifier(x)


# ============================================================
# 6. Load best benchmark model
# ============================================================

model = CNNLSTM().to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_FILE,
        map_location=DEVICE
    )
)

model.eval()

print(
    "\nLoaded best benchmark checkpoint:"
)

print(
    MODEL_FILE
)


# ============================================================
# 7. Test inference
# ============================================================

all_predictions = []
all_targets = []

print(
    "\nRunning inference on test set..."
)


with torch.no_grad():

    for batch_index, (images, targets) in enumerate(
        test_loader
    ):

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        outputs = model(images)

        predictions = outputs.argmax(
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().tolist()
        )

        all_targets.extend(
            targets.tolist()
        )


# ============================================================
# 8. Basic metrics
# ============================================================

accuracy = accuracy_score(
    all_targets,
    all_predictions
)

macro_f1 = f1_score(
    all_targets,
    all_predictions,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    all_targets,
    all_predictions,
    average="weighted",
    zero_division=0
)

macro_precision = precision_score(
    all_targets,
    all_predictions,
    average="macro",
    zero_division=0
)

macro_recall = recall_score(
    all_targets,
    all_predictions,
    average="macro",
    zero_division=0
)


# ============================================================
# 9. Confusion Matrix
# ============================================================

cm = confusion_matrix(
    all_targets,
    all_predictions,
    labels=[0, 1, 2]
)


print("\n" + "=" * 60)

print("CONFUSION MATRIX")

print("=" * 60)

print(
    "             Predicted"
)

print(
    "             C       M       X"
)

for i, name in enumerate(CLASS_NAMES):

    print(
        f"True {name}    "
        f"{cm[i,0]:5d}  "
        f"{cm[i,1]:5d}  "
        f"{cm[i,2]:5d}"
    )


# ============================================================
# 10. Classification report
# ============================================================

report = classification_report(
    all_targets,
    all_predictions,
    target_names=CLASS_NAMES,
    labels=[0, 1, 2],
    digits=4,
    zero_division=0
)

print("\n" + "=" * 60)

print("CLASSIFICATION REPORT")

print("=" * 60)

print(report)


# ============================================================
# 11. Per-class metrics
# ============================================================

precision = precision_score(
    all_targets,
    all_predictions,
    labels=[0, 1, 2],
    average=None,
    zero_division=0
)

recall = recall_score(
    all_targets,
    all_predictions,
    labels=[0, 1, 2],
    average=None,
    zero_division=0
)

f1 = f1_score(
    all_targets,
    all_predictions,
    labels=[0, 1, 2],
    average=None,
    zero_division=0
)


# ============================================================
# 12. Solar-flare metrics
# ============================================================
#
# We calculate these using one-vs-rest for each class.
#
# POD  = TP / (TP + FN)
# FAR  = FP / (TP + FP)
# CSI  = TP / (TP + FN + FP)
#
# TSS  = POD - POFD
#
# HSS  = Heidke Skill Score
#
# Then we calculate macro averages across C/M/X.
# ============================================================

def calculate_binary_metrics(cm, class_index):

    TP = cm[class_index, class_index]

    FN = (
        cm[class_index, :].sum()
        - TP
    )

    FP = (
        cm[:, class_index].sum()
        - TP
    )

    TN = (
        cm.sum()
        - TP
        - FN
        - FP
    )

    # Probability of Detection
    if TP + FN > 0:
        POD = TP / (TP + FN)
    else:
        POD = 0.0


    # False Alarm Ratio
    if TP + FP > 0:
        FAR = FP / (TP + FP)
    else:
        FAR = 0.0


    # Critical Success Index
    if TP + FN + FP > 0:
        CSI = TP / (
            TP + FN + FP
        )
    else:
        CSI = 0.0


    # Probability of False Detection
    if FP + TN > 0:
        POFD = FP / (
            FP + TN
        )
    else:
        POFD = 0.0


    # True Skill Statistic
    TSS = POD - POFD


    # Heidke Skill Score
    total = cm.sum()

    correct = TP + TN

    expected_correct = (
        (
            (TP + FN)
            * (TP + FP)
        )
        +
        (
            (FP + TN)
            * (FN + TN)
        )
    ) / total


    denominator = (
        total
        - expected_correct
    )

    if denominator != 0:

        HSS = (
            correct
            - expected_correct
        ) / denominator

    else:

        HSS = 0.0


    return {
        "POD": POD,
        "FAR": FAR,
        "CSI": CSI,
        "TSS": TSS,
        "HSS": HSS
    }


class_metrics = {}

for i, class_name in enumerate(CLASS_NAMES):

    class_metrics[class_name] = (
        calculate_binary_metrics(
            cm,
            i
        )
    )


# ============================================================
# 13. Print per-class metrics
# ============================================================

print("\n" + "=" * 60)

print("SOLAR FLARE METRICS")

print("=" * 60)


for class_name in CLASS_NAMES:

    metrics = class_metrics[
        class_name
    ]

    print(
        f"\n{class_name} CLASS"
    )

    print(
        f"POD: {metrics['POD']:.4f}"
    )

    print(
        f"FAR: {metrics['FAR']:.4f}"
    )

    print(
        f"CSI: {metrics['CSI']:.4f}"
    )

    print(
        f"TSS: {metrics['TSS']:.4f}"
    )

    print(
        f"HSS: {metrics['HSS']:.4f}"
    )


# ============================================================
# 14. Macro-average solar metrics
# ============================================================

macro_POD = sum(
    class_metrics[c]["POD"]
    for c in CLASS_NAMES
) / NUM_CLASSES

macro_FAR = sum(
    class_metrics[c]["FAR"]
    for c in CLASS_NAMES
) / NUM_CLASSES

macro_CSI = sum(
    class_metrics[c]["CSI"]
    for c in CLASS_NAMES
) / NUM_CLASSES

macro_TSS = sum(
    class_metrics[c]["TSS"]
    for c in CLASS_NAMES
) / NUM_CLASSES

macro_HSS = sum(
    class_metrics[c]["HSS"]
    for c in CLASS_NAMES
) / NUM_CLASSES


print("\n" + "=" * 60)

print("MACRO-AVERAGED SOLAR METRICS")

print("=" * 60)

print(
    f"POD: {macro_POD:.4f}"
)

print(
    f"FAR: {macro_FAR:.4f}"
)

print(
    f"CSI: {macro_CSI:.4f}"
)

print(
    f"TSS: {macro_TSS:.4f}"
)

print(
    f"HSS: {macro_HSS:.4f}"
)


# ============================================================
# 15. Overall summary
# ============================================================

print("\n" + "=" * 60)

print("OVERALL TEST RESULTS")

print("=" * 60)

print(
    f"Test Accuracy:      {accuracy:.4f}"
)

print(
    f"Macro Precision:    {macro_precision:.4f}"
)

print(
    f"Macro Recall:       {macro_recall:.4f}"
)

print(
    f"Macro F1:           {macro_f1:.4f}"
)

print(
    f"Weighted F1:        {weighted_f1:.4f}"
)

print(
    f"Macro POD:          {macro_POD:.4f}"
)

print(
    f"Macro FAR:          {macro_FAR:.4f}"
)

print(
    f"Macro CSI:          {macro_CSI:.4f}"
)

print(
    f"Macro TSS:          {macro_TSS:.4f}"
)

print(
    f"Macro HSS:          {macro_HSS:.4f}"
)


# ============================================================
# 16. Save metrics JSON
# ============================================================

results = {

    "model": "CNN-LSTM",

    "checkpoint": MODEL_FILE,

    "test_sequences": len(test_dataset),

    "accuracy": accuracy,

    "macro_precision": macro_precision,

    "macro_recall": macro_recall,

    "macro_f1": macro_f1,

    "weighted_f1": weighted_f1,

    "macro_POD": macro_POD,

    "macro_FAR": macro_FAR,

    "macro_CSI": macro_CSI,

    "macro_TSS": macro_TSS,

    "macro_HSS": macro_HSS,

    "per_class": class_metrics,

    "confusion_matrix": cm.tolist()
}


with open(
    "cnn_lstm_test_results.json",
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


print(
    "\nSaved: cnn_lstm_test_results.json"
)


# ============================================================
# 17. Save classification report
# ============================================================

with open(
    "cnn_lstm_classification_report.txt",
    "w"
) as f:

    f.write(report)

    f.write(
        "\n\nOverall Metrics\n"
    )

    f.write(
        f"Accuracy: {accuracy:.6f}\n"
    )

    f.write(
        f"Macro F1: {macro_f1:.6f}\n"
    )

    f.write(
        f"Weighted F1: {weighted_f1:.6f}\n"
    )

    f.write(
        f"Macro POD: {macro_POD:.6f}\n"
    )

    f.write(
        f"Macro FAR: {macro_FAR:.6f}\n"
    )

    f.write(
        f"Macro CSI: {macro_CSI:.6f}\n"
    )

    f.write(
        f"Macro TSS: {macro_TSS:.6f}\n"
    )

    f.write(
        f"Macro HSS: {macro_HSS:.6f}\n"
    )


print(
    "Saved: cnn_lstm_classification_report.txt"
)


# ============================================================
# 18. Confusion matrix plot
# ============================================================

plt.figure(
    figsize=(8, 7)
)

plt.imshow(cm)

plt.title(
    "CNN-LSTM Test Confusion Matrix"
)

plt.xlabel(
    "Predicted label"
)

plt.ylabel(
    "True label"
)

plt.xticks(
    range(NUM_CLASSES),
    CLASS_NAMES
)

plt.yticks(
    range(NUM_CLASSES),
    CLASS_NAMES
)


for i in range(NUM_CLASSES):

    for j in range(NUM_CLASSES):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )


plt.colorbar()

plt.tight_layout()

plt.savefig(
    "cnn_lstm_confusion_matrix.png",
    dpi=200
)

plt.close()


print(
    "Saved: cnn_lstm_confusion_matrix.png"
)


# ============================================================
# 19. Final
# ============================================================

print("\n" + "=" * 60)

print("EVALUATION COMPLETE")

print("=" * 60)
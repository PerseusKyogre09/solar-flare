import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


DATA_DIR = Path("data/images/Lat60_Lon60_Nans0_png_224")
LABEL_FILE = Path("data/C1.0_24hr_224_png_Labels.txt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load labels
labels = {}

with open(LABEL_FILE) as f:
    for line in f:
        filename, flare = line.strip().split(",", 1)

        if flare.startswith("C"):
            labels[filename] = 0
        elif flare.startswith("M"):
            labels[filename] = 1
        elif flare.startswith("X"):
            labels[filename] = 2


# Find images
samples = []

for image_path in DATA_DIR.rglob("*.png"):
    filename = image_path.name

    if filename in labels:
        samples.append((image_path, labels[filename]))


# Same split as train.py
train_samples, val_samples = train_test_split(
    samples,
    test_size=0.2,
    random_state=42,
    stratify=[label for _, label in samples]
)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


class SolarDataset(Dataset):

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("L")
        image = transform(image)

        return image, label


val_dataset = SolarDataset(val_samples)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False
)


# Model
class TinyCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Linear(64, 3)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


model = TinyCNN().to(DEVICE)

model.load_state_dict(
    torch.load(
        "solar_flare_tinycnn.pth",
        map_location=DEVICE,
        weights_only=True
    )
)

model.eval()


# Evaluate
all_predictions = []
all_targets = []

with torch.no_grad():

    for images, targets in val_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        all_predictions.extend(predictions.cpu().numpy())
        all_targets.extend(targets.numpy())


# Confusion matrix
cm = confusion_matrix(
    all_targets,
    all_predictions
)

print("\nConfusion Matrix:")
print(cm)


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["C", "M", "X"]
)

disp.plot()

plt.title("Solar Flare Classification - Tiny CNN")

plt.savefig(
    "confusion_matrix.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print("\nSaved confusion_matrix.png")


import csv
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split


# =========================
# 1. Configuration
# =========================

DATA_DIR = Path("data/images/Lat60_Lon60_Nans0_png_224")
LABEL_FILE = Path("data/C1.0_24hr_224_png_Labels.txt")

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

train_losses = []
train_accuracies = []
val_accuracies = []


# =========================
# 2. Load labels
# =========================

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


# =========================
# 3. Find our 300 images
# =========================

samples = []

for image_path in DATA_DIR.rglob("*.png"):

    filename = image_path.name

    if filename in labels:
        samples.append((image_path, labels[filename]))


print("Total images:", len(samples))


# =========================
# 4. Train / validation split
# =========================

random.seed(42)

train_samples, val_samples = train_test_split(
    samples,
    test_size=0.2,
    random_state=42,
    stratify=[label for _, label in samples]
)

print("Training images:", len(train_samples))
print("Validation images:", len(val_samples))


# =========================
# 5. Image transformations
# =========================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# =========================
# 6. Dataset
# =========================

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


train_dataset = SolarDataset(train_samples)
val_dataset = SolarDataset(val_samples)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# =========================
# 7. Tiny CNN
# =========================

class TinyCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Linear(64, 3)


    def forward(self, x):

        x = self.features(x)

        x = x.view(x.size(0), -1)

        return self.classifier(x)


model = TinyCNN().to(DEVICE)


# =========================
# 8. Loss + optimizer
# =========================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# =========================
# 9. Training
# =========================

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, targets in train_loader:

        images = images.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, targets)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        correct += (predictions == targets).sum().item()

        total += targets.size(0)


    train_accuracy = correct / total


    # =========================
    # Validation
    # =========================

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, targets in val_loader:

            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            val_correct += (predictions == targets).sum().item()

            val_total += targets.size(0)


    val_accuracy = val_correct / val_total

    epoch_loss = total_loss / len(train_loader)

    train_losses.append(epoch_loss)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {epoch_loss:.4f} | "
        f"Train Acc: {train_accuracy:.3f} | "
        f"Val Acc: {val_accuracy:.3f}"
    )


# =========================
# 10. Save model
# =========================

torch.save(
    model.state_dict(),
    "solar_flare_tinycnn.pth"
)

print("\nModel saved to solar_flare_tinycnn.pth")

# =========================
# 11. Save training graphs
# =========================

import matplotlib.pyplot as plt

# Training loss
plt.figure()

plt.plot(train_losses)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")

plt.savefig(
    "training_loss.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# Training and validation accuracy
plt.figure()

plt.plot(train_accuracies, label="Training Accuracy")
plt.plot(val_accuracies, label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")

plt.legend()

plt.savefig(
    "training_accuracy.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("Training graphs saved.")

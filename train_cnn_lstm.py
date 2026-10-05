import os
import random
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from torchvision import transforms

from sklearn.metrics import (
    f1_score,
    recall_score,
    confusion_matrix
)


class FocalLoss(nn.Module):
    def __init__(self, gamma=0.8, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        ce = F.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        focal_term = (1.0 - pt).pow(self.gamma)
        loss = ce * focal_term

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def compute_class_weights(class_counts):
    raw = 1.0 / torch.sqrt(class_counts.clamp_min(1.0))
    raw = raw / raw.mean()
    weights = torch.clamp(raw, min=0.35, max=2.5)
    weights = weights / weights.mean()
    return weights


def choose_x_threshold(probs, targets):
    x_probs = probs[:, 2]
    best_threshold = 0.5
    best_score = -1.0

    for threshold in np.linspace(0.1, 0.9, 81):
        pred = np.where(x_probs.cpu().numpy() >= threshold, 2, np.argmax(probs.cpu().numpy(), axis=1))
        macro_f1 = f1_score(targets, pred, average="macro", zero_division=0)
        recalls = recall_score(targets, pred, average=None, labels=[0, 1, 2], zero_division=0)
        balanced = recalls.mean()
        score = 0.7 * macro_f1 + 0.3 * balanced

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold


# ============================================================
# 1. Reproducibility
# ============================================================

def main():
    SEED = 42

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

    # Deterministic behaviour where supported.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    # ============================================================
    # 2. Configuration
    # ============================================================

    DATA_DIR = Path(
        "data/images/Lat60_Lon60_Nans0_png_224"
    )

    SEQUENCE_LENGTH = 10

    BATCH_SIZE = 8

    MAX_EPOCHS = int(os.environ.get("MAX_EPOCHS", 20))

    HEAD_LR = 0.001
    BACKBONE_LR = 0.0001
    LEARNING_RATE = HEAD_LR

    WEIGHT_DECAY = 1e-4

    PATIENCE = int(os.environ.get("PATIENCE", 8))

    GRADIENT_CLIP = 1.0

    HEAD_WARMUP_EPOCHS = 3

    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("CNN-LSTM TRAINING")
    print("=" * 60)

    print("Random seed:", SEED)
    print("Device:", DEVICE)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))


    # ============================================================
    # 3. Dataset
    # ============================================================

    class SolarSequenceDataset(Dataset):

        def __init__(self, csv_file):

            self.df = pd.read_csv(csv_file)

            frame_columns = [
                f"frame_{i}"
                for i in range(SEQUENCE_LENGTH)
            ]

            available = self.df[frame_columns].apply(
                lambda row: all(
                    (DATA_DIR / row[column]).is_file()
                    for column in frame_columns
                ),
                axis=1
            )

            missing_sequences = int((~available).sum())

            if missing_sequences:
                print(
                    f"Skipping {missing_sequences} sequences with missing frames "
                    f"from {csv_file}"
                )

            self.df = self.df.loc[available].reset_index(drop=True)

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

            # [sequence, channels, height, width]
            images = torch.stack(images)

            label = self.class_map[
                row["target"]
            ]

            return images, label


    # ============================================================
    # 4. Load datasets
    # ============================================================

    train_dataset = SolarSequenceDataset(
        "data/Train_sequences.csv"
    )

    val_dataset = SolarSequenceDataset(
        "data/Validation_sequences.csv"
    )

    test_dataset = SolarSequenceDataset(
        "data/Test_sequences.csv"
    )

    print("\nDataset sizes:")
    print("Train:", len(train_dataset))
    print("Validation:", len(val_dataset))
    print("Test:", len(test_dataset))


    # ============================================================
    # 5. DataLoaders
    # ============================================================

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_class_ids = torch.tensor(
        [
            train_dataset.class_map[target]
            for target in train_dataset.df["target"]
        ],
        dtype=torch.long
    )

    class_counts = torch.bincount(
        train_class_ids,
        minlength=3
    ).float()

    # Increase minority-class pressure without discarding C examples.
    sample_weights = 1.0 / torch.sqrt(class_counts[train_class_ids])
    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True,
        generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        generator=generator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )


    # ============================================================
    # 6. CNN Encoder
    # ============================================================

    class CNNEncoder(nn.Module):

        def __init__(self):

            super().__init__()

            self.features = nn.Sequential(

                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )

        def forward(self, x):
            x = self.features(x)
            return x.view(x.size(0), -1)


    # ============================================================
    # 7. CNN-LSTM
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
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 3)
            )

        def forward(self, x):
            batch_size = x.size(0)
            sequence_length = x.size(1)

            x = x.view(batch_size * sequence_length, 1, 224, 224)
            x = self.cnn(x)
            x = x.view(batch_size, sequence_length, 64)
            x, _ = self.lstm(x)
            x = x[:, -1, :]
            return self.classifier(x)


    model = CNNLSTM().to(DEVICE)

    print("\nModel parameters:", sum(p.numel() for p in model.parameters()))


    # ============================================================
    # 8. Loss
    # ============================================================

    print("\nTraining class counts:", class_counts.tolist())

    class_weights = compute_class_weights(class_counts).to(DEVICE)
    print("\nClass weights:", class_weights.cpu().tolist())

    criterion = FocalLoss(
        gamma=0.8,
        alpha=class_weights
    )


    # ============================================================
    # 9. Optimizer
    # ============================================================

    optimizer = torch.optim.AdamW([
        {"params": model.classifier.parameters(), "lr": HEAD_LR},
        {"params": list(model.cnn.parameters()) + list(model.lstm.parameters()), "lr": BACKBONE_LR},
    ], weight_decay=WEIGHT_DECAY)


    # ============================================================
    # 10. Learning-rate scheduler
    # ============================================================

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )


    # ============================================================
    # 11. Training history
    # ============================================================

    history = []
    best_score = -1.0
    best_epoch = 0
    best_x_recall = 0.0
    best_balanced_accuracy = 0.0
    epochs_without_improvement = 0


    # ============================================================
    # 12. Training
    # ============================================================

    for epoch in range(MAX_EPOCHS):
        print("\n" + "=" * 60)
        print(f"Epoch {epoch + 1}/{MAX_EPOCHS}")
        print("=" * 60)

        if epoch < HEAD_WARMUP_EPOCHS:
            for p in model.cnn.parameters():
                p.requires_grad = False
            for p in model.lstm.parameters():
                p.requires_grad = False
        else:
            for p in model.cnn.parameters():
                p.requires_grad = True
            for p in model.lstm.parameters():
                p.requires_grad = True

        model.train()
        running_loss = 0.0
        running_x_penalty = 0.0
        train_correct = 0
        train_total = 0

        for images, targets in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            base_loss = criterion(outputs, targets)
            
            final_loss = base_loss
            x_penalty = 0.0
            if epoch >= HEAD_WARMUP_EPOCHS and best_x_recall > 0.2:
                x_probs = torch.softmax(outputs, dim=1)[:, 2]
                x_pred_count = (x_probs > 0.5).sum().item()
                if x_pred_count == 0:
                    x_penalty_val = 0.2 * max(best_x_recall, 0.3)
                    x_penalty = x_penalty_val
                    final_loss = base_loss + x_penalty_val
            
            final_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()

            running_loss += final_loss.item()
            running_x_penalty += float(x_penalty)
            predictions = outputs.argmax(dim=1)
            train_correct += (predictions == targets).sum().item()
            train_total += targets.size(0)

        train_loss = running_loss / len(train_loader)
        train_x_penalty = running_x_penalty / len(train_loader)
        train_accuracy = train_correct / train_total

        model.eval()

        val_correct = 0
        val_total = 0
        val_predictions = []
        val_targets = []
        val_probs_all = []

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)

                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                val_probs_all.append(probs.cpu())
                val_targets.extend(targets.cpu().tolist())
        
        val_probs_all = torch.cat(val_probs_all, dim=0)
        x_threshold = choose_x_threshold(val_probs_all, val_targets)
        
        predictions = torch.where(
            val_probs_all[:, 2] >= x_threshold,
            torch.full_like(torch.tensor(val_targets), 2),
            torch.argmax(val_probs_all, dim=1)
        )
        val_predictions = predictions.tolist()
        val_correct = (predictions == torch.tensor(val_targets)).sum().item()
        val_total = len(val_targets)

        val_accuracy = val_correct / val_total

        macro_f1 = f1_score(val_targets, val_predictions, average="macro", zero_division=0)
        per_class_recall = recall_score(
            val_targets,
            val_predictions,
            average=None,
            labels=[0, 1, 2],
            zero_division=0
        )

        c_recall = per_class_recall[0]
        m_recall = per_class_recall[1]
        x_recall = per_class_recall[2]
        balanced_accuracy = per_class_recall.mean()
        score = 0.7 * macro_f1 + 0.3 * balanced_accuracy
        
        if x_recall > best_x_recall or (x_recall == best_x_recall and balanced_accuracy > best_balanced_accuracy):
            best_x_recall = x_recall
            best_balanced_accuracy = balanced_accuracy

        cm = confusion_matrix(val_targets, val_predictions, labels=[0, 1, 2])

        scheduler.step(score)
        current_lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch + 1,
            "loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "macro_f1": macro_f1,
            "balanced_accuracy": balanced_accuracy,
            "score": score,
            "C_recall": c_recall,
            "M_recall": m_recall,
            "X_recall": x_recall,
            "learning_rate": current_lr,
        })

        print(f"Loss:        {train_loss:.4f}")
        if train_x_penalty > 0:
            print(f"X Penalty:   {train_x_penalty:.4f}")
        print(f"Train Acc:   {train_accuracy:.4f}")
        print(f"Val Acc:     {val_accuracy:.4f}")
        print(f"Macro F1:    {macro_f1:.4f}")
        print(f"Balanced Rec: {balanced_accuracy:.4f}")
        print(f"C Recall:    {c_recall:.4f}")
        print(f"M Recall:    {m_recall:.4f}")
        print(f"X Recall:    {x_recall:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        print(f"Head warmup:  {epoch < HEAD_WARMUP_EPOCHS}")
        print("\nValidation Confusion Matrix:")
        print(cm)

        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "cnn_lstm_best.pth")
            print("\n✓ New best model saved.")
            print(f"  Best Score: {best_score:.4f}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

    torch.save(model.state_dict(), "cnn_lstm_final.pth")

    history_df = pd.DataFrame(history)
    history_df.to_csv("cnn_lstm_training_history.csv", index=False)

    config = {
        "seed": SEED,
        "sequence_length": SEQUENCE_LENGTH,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "head_learning_rate": HEAD_LR,
        "backbone_learning_rate": BACKBONE_LR,
        "weight_decay": WEIGHT_DECAY,
        "early_stopping_patience": PATIENCE,
        "gradient_clip": GRADIENT_CLIP,
        "optimizer": "AdamW",
        "loss": "FocalLoss + inverse-sqrt class sampling + class weights",
        "best_checkpoint_metric": "Validation score = 0.7 * Macro F1 + 0.3 * balanced recall",
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
    }

    with open("cnn_lstm_config.txt", "w") as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")


if __name__ == "__main__":
    main()
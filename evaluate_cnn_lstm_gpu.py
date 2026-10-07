"""Run isolated GPU inference for the saved CNN-LSTM test checkpoint."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, matthews_corrcoef
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


ROOT = Path(__file__).resolve().parent
SEQUENCE_LENGTH = 10
CLASS_NAMES = ("C", "M", "X")


class SolarSequenceDataset(Dataset):
    def __init__(self, csv_path: Path, image_root: Path):
        self.frame = pd.read_csv(csv_path)
        self.image_root = image_root
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        missing_columns = {"ar", "target", *(f"frame_{index}" for index in range(SEQUENCE_LENGTH))} - set(self.frame.columns)
        if missing_columns:
            raise ValueError(f"Sequence CSV lacks columns: {sorted(missing_columns)}")
        if not self.frame.target.isin(CLASS_NAMES).all():
            raise ValueError("Test sequence labels must be C, M, or X")
        if self.frame[[f"frame_{index}" for index in range(SEQUENCE_LENGTH)]].isna().any().any():
            raise ValueError("Test sequence CSV contains missing frame paths")

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        frames = []
        for frame_index in range(SEQUENCE_LENGTH):
            path = self.image_root / str(row[f"frame_{frame_index}"])
            with Image.open(path) as image:
                frames.append(self.transform(image.convert("L")))
        images = torch.stack(frames)
        label = CLASS_NAMES.index(row.target)
        return images, label


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
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, images):
        features = self.features(images)
        return features.view(features.size(0), -1)


class CNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = CNNEncoder()
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=1, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3),
        )

    def forward(self, images):
        batch_size, sequence_length = images.shape[:2]
        images = images.view(batch_size * sequence_length, 1, 224, 224)
        features = self.cnn(images).view(batch_size, sequence_length, 64)
        sequence, _ = self.lstm(features)
        return self.classifier(sequence[:, -1, :])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error("batch size must be positive and workers non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm/CUDA device is unavailable; refusing CPU fallback")

    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(0)
    checkpoint_path = ROOT / "cnn_lstm_best.pth"
    sequence_path = ROOT / "data/Test_sequences.csv"
    image_root = ROOT / "data/images/Lat60_Lon60_Nans0_png_224"
    output_dir = ROOT / "experiments/cnn_lstm/multiclass"
    prediction_path = output_dir / "predictions/test_predictions.csv"
    metrics_path = output_dir / "metrics/gpu_test_metrics.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = SolarSequenceDataset(sequence_path, image_root)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
    )
    model = CNNLSTM().to(device)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()

    probability_batches = []
    target_batches = []
    start = time.perf_counter()
    with torch.inference_mode():
        for batch_index, (images, targets) in enumerate(loader, start=1):
            images = images.to(device)
            logits = model(images)
            probability_batches.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            target_batches.append(targets.numpy())
            if batch_index % 50 == 0:
                torch.cuda.synchronize()
                print(f"inference batches: {batch_index}/{len(loader)}")
    torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - start

    probabilities = np.concatenate(probability_batches)
    targets = np.concatenate(target_batches)
    predictions = probabilities.argmax(axis=1)
    if len(predictions) != len(dataset) or not np.isfinite(probabilities).all():
        raise RuntimeError("Inference output row count or probabilities are invalid")

    result_frame = dataset.frame.copy()
    result_frame["true_label_id"] = targets
    result_frame["predicted_label"] = [CLASS_NAMES[index] for index in predictions]
    for index, class_name in enumerate(CLASS_NAMES):
        result_frame[f"probability_{class_name}"] = probabilities[:, index]
    result_frame.to_csv(prediction_path, index=False)

    matrix = confusion_matrix(targets, predictions, labels=[0, 1, 2])
    report = classification_report(
        targets,
        predictions,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "model": "CNN-LSTM",
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "split": "Test",
        "device": device_name,
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "batch_size": args.batch_size,
        "test_sequences": int(len(targets)),
        "elapsed_inference_seconds": elapsed_seconds,
        "accuracy": float(accuracy_score(targets, predictions)),
        "multiclass_mcc": float(matthews_corrcoef(targets, predictions)),
        "class_order": list(CLASS_NAMES),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
        "probabilities_saved": True,
        "predictions_csv": str(prediction_path.relative_to(ROOT)),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    print(f"GPU inference complete: {len(dataset)} sequences in {elapsed_seconds:.1f}s on {device_name}")
    print(f"Predictions: {prediction_path.relative_to(ROOT)}")
    print(f"Metrics: {metrics_path.relative_to(ROOT)}")
    print("Confusion matrix:")
    print(matrix)


if __name__ == "__main__":
    main()
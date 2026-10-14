"""Small local diagnostic for ROCm/CUDA and real magnetogram tensors.

This does not train a research run. It performs one forward/backward/optimizer
step and saves a temporary checkpoint plus probabilities under results/diagnostics.
"""
from pathlib import Path
import json, time
import numpy as np
from PIL import Image
import torch
from torch import nn

def main():
    out = Path("results/diagnostics"); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = sorted(Path("data").rglob("*.png"))
    if not images: raise FileNotFoundError("No PNG magnetograms found below data/")
    image_path = images[0]
    arr = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).repeat(2, 1, 1, 1).to(device)
    y = torch.tensor([0, 1], dtype=torch.long, device=device)
    model = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 2)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    t0 = time.perf_counter(); logits = model(x); loss = nn.CrossEntropyLoss()(logits, y); loss.backward(); opt.step(); elapsed = time.perf_counter() - t0
    ckpt = out / "smoke_checkpoint.pt"; torch.save(model.state_dict(), ckpt)
    restored = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 2)).to(device)
    restored.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    restored.eval()
    with torch.no_grad(): reloaded_logits = restored(x)
    reload_max_abs_diff = float((reloaded_logits - logits.detach()).abs().max().cpu())
    probs = torch.softmax(logits.detach().cpu(), dim=1).tolist()
    report = {"image": str(image_path), "image_shape": list(arr.shape), "device": str(device), "torch": torch.__version__, "hip": torch.version.hip, "cuda_build": torch.version.cuda, "available": torch.cuda.is_available(), "device_count": torch.cuda.device_count(), "loss": float(loss.detach().cpu()), "step_seconds": elapsed, "checkpoint": str(ckpt), "reload_max_abs_logit_difference": reload_max_abs_diff, "probabilities": probs}
    (out / "accelerator_smoke_test.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == "__main__": main()

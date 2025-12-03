"""
Lightweight sanity check for the CV starter.

Loads CIFAR-10 using the configured transforms, runs a single forward/backward
pass, and stores metrics under outputs/smoke_metrics.json. Helpful for CPU-only
validation before longer GPU runs (e.g., on Colab/Kaggle T4).
"""

from pathlib import Path
import torch
import torch.nn as nn

from utils import load_yaml, set_seed, get_device, save_json
from data import build_dataloaders
from model import build_model


def run_smoke(cfg_path: str = "configs/cv_cifar10_fast.yaml") -> Path:
    cfg = load_yaml(cfg_path)
    set_seed(cfg["seed"])
    device = get_device()

    train_loader, _, num_classes, _ = build_dataloaders(cfg)
    model = build_model(cfg, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    model.train()
    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device)

    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "smoke_metrics.json"
    save_json(
        {
            "loss": float(loss.item()),
            "batch_size": int(x.size(0)),
            "num_classes": num_classes,
            "device": str(device),
        },
        out_path,
    )
    return out_path


if __name__ == "__main__":
    path = run_smoke()
    print(f"Smoke check succeeded. Metrics written to {path}.")

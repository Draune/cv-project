"""Command-line entrypoint for training the CV starter model.

The script ties all pieces together: it loads hyperparameters from YAML, builds
the dataloaders, instantiates a pretrained backbone, and handles the training
loop with logging, learning-rate scheduling, early stopping, and checkpointing.
Because everything lives behind configuration flags, students can iterate on
the experiment design without modifying the core code.
"""

import argparse, csv
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm

from utils import set_seed, get_device, load_yaml, save_checkpoint, save_json
from data import build_dataloaders
from model import build_model

def train_one_epoch(model, loader, crit, opt, device):
    """Run one training pass and return the average loss.

    Parameters
    ----------
    model : torch.nn.Module
        Network being optimised.
    loader : DataLoader
        Yields mini-batches for training.
    crit : torch.nn.Module
        Loss function (cross-entropy for classification).
    opt : torch.optim.Optimizer
        Optimiser handling parameter updates.
    device : torch.device
        Computation device chosen via `get_device`.
    """
    model.train()
    loss_sum, n = 0.0, 0
    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = crit(logits, y)
        loss.backward()
        opt.step()
        loss_sum += loss.item() * y.size(0); n += y.size(0)
    return loss_sum / n

@torch.no_grad()
def evaluate(model, loader, crit, device, num_classes):
    """Evaluate the model on a validation loader and return loss + accuracy."""
    model.eval()
    acc = MulticlassAccuracy(num_classes=num_classes).to(device)
    loss_sum, n = 0.0, 0
    for x, y in tqdm(loader, desc="val", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = crit(logits, y)
        loss_sum += loss.item() * y.size(0); n += y.size(0)
        acc.update(logits, y)
    return loss_sum / n, acc.compute().item()

def main(cfg_path):
    """Read configuration from disk and execute the full training workflow."""
    cfg = load_yaml(cfg_path)
    set_seed(cfg["seed"])
    device = get_device()

    out = Path(cfg["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    (out / "log.csv").write_text("epoch,train_loss,val_loss,val_acc\n")

    train_loader, val_loader, num_classes, classes = build_dataloaders(cfg)
    model = build_model(cfg, num_classes).to(device)
    crit = nn.CrossEntropyLoss()

    params = [p for p in model.parameters() if p.requires_grad]
    if cfg["train"]["optimizer"].lower() == "sgd":
        opt = SGD(params, lr=cfg["train"]["lr"], momentum=cfg["train"]["momentum"], weight_decay=cfg["train"]["weight_decay"])
    else:
        opt = AdamW(params, lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    if cfg["train"]["scheduler"] == "cosine":
        sch = CosineAnnealingLR(opt, T_max=cfg["train"]["t_max"])
    else:
        sch = None

    best_acc, best_path = 0.0, out / "best.pt"
    patience, waited = cfg["early_stopping"]["patience"], 0

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        tr_loss = train_one_epoch(model, train_loader, crit, opt, device)
        val_loss, val_acc = evaluate(model, val_loader, crit, device, num_classes)
        if sch: sch.step()

        with open(out / "log.csv", "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{tr_loss:.4f}", f"{val_loss:.4f}", f"{val_acc:.4f}"])

        if val_acc > best_acc + cfg["early_stopping"]["min_delta"]:
            best_acc, waited = val_acc, 0
            save_checkpoint(model, best_path)
        else:
            waited += 1
            if waited > patience:
                break

    save_json({"best_val_acc": best_acc, "classes": classes}, out / "metrics.json")
    print(f"Done. Best val acc: {best_acc:.4f}. Checkpoint: {best_path}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/cv_cifar10.yaml")
    args = ap.parse_args()
    main(args.config)

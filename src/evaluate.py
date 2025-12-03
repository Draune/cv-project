"""Evaluation CLI for the CV starter.

Given a configuration file and a trained checkpoint, this script rebuilds the
model architecture, runs it on the validation split, and generates a rich set
of artefacts inside `outputs/`:

* `eval.json` with accuracy and macro-level metrics.
* `per_class_metrics.csv` enumerating precision/recall/F1/support per class.
* `confusion_matrix.png` heatmap rendered with seaborn.
* `leaderboard.png` bar chart that visualises per-class precision/recall/F1.
"""

import argparse, json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import classification_report
from torchmetrics.classification import MulticlassAccuracy, ConfusionMatrix

from utils import load_yaml
from data import build_dataloaders
from model import build_model

@torch.no_grad()
def main(cfg_path, ckpt_path):
    """Evaluate a checkpoint using configuration-driven dataloaders.

    Parameters
    ----------
    cfg_path : str | Path
        YAML file describing data/model/training options.
    ckpt_path : str | Path
        Location of the `.pt` state dict produced by `train.py`.
    """
    cfg = load_yaml(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader, num_classes, classes = build_dataloaders(cfg)

    model = build_model(cfg, num_classes)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()

    acc_metric = MulticlassAccuracy(num_classes=num_classes).to(device)
    cm_metric = ConfusionMatrix(num_classes=num_classes).to(device)

    preds_all, targets_all = [], []

    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = torch.argmax(logits, dim=1)

        acc_metric.update(logits, y)
        cm_metric.update(logits, y)

        preds_all.append(preds.cpu())
        targets_all.append(y.cpu())

    acc = acc_metric.compute().item()
    cm = cm_metric.compute().cpu().numpy()
    preds_np = torch.cat(preds_all).numpy()
    targets_np = torch.cat(targets_all).numpy()

    report = classification_report(
        targets_np,
        preds_np,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    per_class_df = pd.DataFrame(report).transpose().loc[classes]
    macro_avg = report["macro avg"]
    weighted_avg = report["weighted avg"]

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "eval.json", "w") as f:
        json.dump(
            {
                "val_acc": acc,
                "macro_precision": macro_avg["precision"],
                "macro_recall": macro_avg["recall"],
                "macro_f1": macro_avg["f1-score"],
                "weighted_precision": weighted_avg["precision"],
                "weighted_recall": weighted_avg["recall"],
                "weighted_f1": weighted_avg["f1-score"],
                "accuracy": report["accuracy"],
                "num_samples": int(len(targets_np)),
                "classes": classes,
            },
            f,
            indent=2,
        )

    per_class_df.to_csv(out / "per_class_metrics.csv", float_format="%.4f")

    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar_kws={"shrink": 0.75},
        ax=ax,
    )
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    leaderboard_df = (
        per_class_df[["precision", "recall", "f1-score"]]
        .reset_index()
        .rename(columns={"index": "class"})
        .melt(id_vars="class", var_name="metric", value_name="score")
    )
    fig, ax = plt.subplots(figsize=(10, max(5, len(classes) * 0.4)))
    sns.barplot(
        data=leaderboard_df,
        x="score",
        y="class",
        hue="metric",
        palette="viridis",
        ax=ax,
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("Score")
    ax.set_ylabel("Class")
    ax.set_title("Per-class Metrics Leaderboard")
    ax.legend(title="Metric", loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "leaderboard.png", dpi=200)
    plt.close(fig)

    print(
        f"Accuracy: {acc:.4f}. Saved eval.json, per_class_metrics.csv, "
        f"confusion_matrix.png, and leaderboard.png in {out}"
    )

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/cv_cifar10.yaml")
    p.add_argument("--ckpt",   type=str, default="outputs/best.pt")
    a = p.parse_args()
    main(a.config, a.ckpt)

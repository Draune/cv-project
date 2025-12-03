"""Data loading helpers used by the CV starter.

The functions here keep all dataset-specific logic in one place so the rest of
the codebase can stay agnostic to whether we are working with CIFAR-10 or a
custom ImageFolder.  Transforms mirror standard transfer-learning recipes: we
resize to the target input size, add basic augmentation for training, and apply
ImageNet normalization so pretrained weights behave as expected.
"""

from pathlib import Path
from typing import Tuple
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def _build_transforms(img_size: int):
    """Build torchvision transforms for training and validation.

    Training gets light augmentation (resize, crop, flip) to encourage
    generalisation, while validation receives deterministic resizing so that
    metrics remain stable across runs.
    """
    train_tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf

def build_dataloaders(cfg) -> Tuple[DataLoader, DataLoader, int, list]:
    """Create dataloaders and metadata according to the configuration dictionary.

    Returns
    -------
    train_loader, val_loader : torch.utils.data.DataLoader
        Iterables ready for the training loop.
    num_classes : int
        Number of unique labels, used to size the classifier head.
    classes : list[str]
        Human-readable class names for downstream reporting.
    """
    img_size = cfg["data"]["img_size"]
    train_tf, val_tf = _build_transforms(img_size)
    root = Path(cfg["data"]["root"])

    if cfg["data"]["dataset"].lower() == "cifar10":
        train_set = datasets.CIFAR10(root=str(root), train=True, download=True, transform=train_tf)
        val_set   = datasets.CIFAR10(root=str(root), train=False, download=True, transform=val_tf)
        classes = train_set.classes
    elif cfg["data"]["dataset"].lower() == "imagefolder":
        full = datasets.ImageFolder(root=str(root), transform=train_tf)
        val_ratio = float(cfg["data"]["val_split"])
        n_val = max(1, int(len(full) * val_ratio))
        n_train = len(full) - n_val
        train_set, val_set = random_split(full, [n_train, n_val])
        # apply val transforms to the validation subset
        val_set.dataset.transform = val_tf
        classes = full.classes
    else:
        raise ValueError("Unknown dataset type")

    train_loader = DataLoader(train_set, batch_size=cfg["data"]["batch_size"],
                              shuffle=True, num_workers=cfg["data"]["num_workers"], pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=cfg["data"]["batch_size"],
                              shuffle=False, num_workers=cfg["data"]["num_workers"], pin_memory=True)
    num_classes = len(classes)
    return train_loader, val_loader, num_classes, classes

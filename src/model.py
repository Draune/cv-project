"""Model factory used by the CV starter.

The goal is to hide torchvision specifics behind a simple configuration
interface so beginners only have to change YAML values to swap architectures or
toggle backbone freezing.
"""

import torch.nn as nn
import torchvision.models as tv

def build_model(cfg, num_classes: int) -> nn.Module:
    """Construct a classification network according to the config dictionary."""
    arch = cfg["model"]["arch"].lower()
    pretrained = bool(cfg["model"]["pretrained"])

    if arch == "resnet18":
        m = tv.resnet18(weights=tv.ResNet18_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif arch == "resnet34":
        m = tv.resnet34(weights=tv.ResNet34_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    else:
        raise ValueError("Unsupported arch")

    if cfg["model"]["freeze_backbone"]:
        for n, p in m.named_parameters():
            if not n.startswith("fc."):
                p.requires_grad = False
    return m

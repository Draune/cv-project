"""Shared utilities: seeding, device helpers, and lightweight IO wrappers."""

import json, random
from pathlib import Path
import numpy as np
import torch
import yaml

def set_seed(seed: int):
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    """Return a CUDA device when available, otherwise default to CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_yaml(path):
    """Parse a YAML file and return its contents as a Python dictionary."""
    with open(path, "r") as f: return yaml.safe_load(f)

def save_json(obj, path):
    """Serialize `obj` to JSON, creating parent directories as required."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f: json.dump(obj, f, indent=2)

def save_checkpoint(model, path):
    """Persist a model state dict to disk, ensuring parent directories exist."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    import torch
    torch.save(model.state_dict(), path)

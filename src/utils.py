"""Shared configuration, reproducibility and checkpoint helpers."""
from __future__ import annotations

import random
import json
from pathlib import Path

import numpy as np


def load_config(path: str | Path = "config.yaml") -> dict:
    # JSON is a strict subset of YAML, so config.yaml remains portable without a parser dependency.
    return json.loads(Path(path).read_text(encoding="utf-8"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name: str = "auto"):
    import torch
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot detect a CUDA device")
    return torch.device(name)


def save_checkpoint(path, *, epoch, generator, discriminator, optimizer_g, optimizer_d,
                    best_val_loss, config):
    import torch
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "optimizer_g": optimizer_g.state_dict(), "optimizer_d": optimizer_d.state_dict(),
                "best_val_loss": best_val_loss, "config": config}, path)


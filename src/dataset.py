"""Lazy sequence windows built independently within each stored trip."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TripWindowDataset(Dataset):
    def __init__(self, processed_dir: str | Path, split: str, sequence_length: int = 256,
                 stride: int = 1, max_sequences: int | None = None):
        self.root = Path(processed_dir)
        split_data = json.loads((self.root / "splits.json").read_text(encoding="utf-8"))
        if split not in split_data:
            raise ValueError(f"Unknown split {split!r}; choose train, validation, or test")
        self.sequence_length = sequence_length
        self.index: list[tuple[str, int]] = []
        for trip_id in split_data[split]:
            data = np.load(self.root / "trips" / f"{trip_id}.npz")
            x, y = data["x"], data["y"]
            valid = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
            for start in range(0, len(valid) - sequence_length + 1, stride):
                if valid[start:start + sequence_length].all():
                    self.index.append((trip_id, start))
        if max_sequences is not None:
            self.index = self.index[:max_sequences]
        self._cache_id = None
        self._cache = None

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int):
        trip_id, start = self.index[item]
        if trip_id != self._cache_id:
            self._cache = np.load(self.root / "trips" / f"{trip_id}.npz")
            self._cache_id = trip_id
        end = start + self.sequence_length
        x = torch.from_numpy(self._cache["x"][start:end].T.copy())
        y = torch.from_numpy(self._cache["y"][start:end].T.copy())
        return x, y


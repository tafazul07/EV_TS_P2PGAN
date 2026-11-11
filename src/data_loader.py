"""Read trip CSVs safely without joining trip boundaries."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def read_trip(path: str | Path, *, encoding: str = "cp1252") -> pd.DataFrame:
    """Load one semicolon-delimited measurement trip with physical units in names."""
    path = Path(path)
    frame = pd.read_csv(path, sep=";", encoding=encoding, na_values=["-", "NaN", "nan", "NA"])
    original_columns = [str(name) for name in frame.columns]
    normalized = [name.strip() for name in original_columns]
    # Preserve whitespace where stripping would collapse two distinct source channels.
    counts = {name: normalized.count(name) for name in set(normalized)}
    normalized = [original if counts[clean] > 1 else clean
                  for original, clean in zip(original_columns, normalized)]
    aliases = {"Velocity [km/h]]]": "Velocity [km/h]"}
    applied = {}
    for index, name in enumerate(normalized):
        if name in aliases:
            applied[name] = aliases[name]
            normalized[index] = aliases[name]
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{path.name}: header normalization creates duplicate column names")
    frame.columns = normalized
    frame.attrs["header_aliases"] = applied
    if "Time [s]" not in frame.columns:
        raise ValueError(f"{path.name}: missing required 'Time [s]' column")
    frame.insert(0, "trip_id", path.stem)
    return frame


def list_trip_files(data_dir: str | Path) -> list[Path]:
    """List only raw TripA/TripB CSVs (not generated inventory CSV files)."""
    root = Path(data_dir)
    return sorted(p for p in root.glob("Trip*.csv") if p.stem[:5] in {"TripA", "TripB"} and p.stem[5:].isdigit())


def validate_features(
    frame: pd.DataFrame, inputs: Iterable[str], targets: Iterable[str], *, trip_id: str = "unknown"
) -> list[str]:
    """Return missing required columns, preserving the source rather than fabricating them."""
    required = [*inputs, *targets]
    return [column for column in required if column not in frame.columns]


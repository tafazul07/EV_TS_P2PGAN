"""Trip-safe smoothing, reproducible trip splits, train-only scaling and windows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .data_loader import list_trip_files, read_trip, validate_features
from .utils import load_config


def split_trip_ids(trip_ids: list[str], seed: int, train: float, validation: float) -> dict[str, list[str]]:
    """Split by trip within season so no overlapping windows leak across splits."""
    rng = np.random.default_rng(seed)
    split = {"train": [], "validation": [], "test": []}
    for season in ("TripA", "TripB"):
        group = np.array(sorted(x for x in trip_ids if x.startswith(season)))
        rng.shuffle(group)
        n_train = int(round(len(group) * train))
        n_validation = int(round(len(group) * validation))
        # Keep every season represented in each split where its sample count allows.
        n_train = min(max(1, n_train), len(group) - 2)
        n_validation = min(max(1, n_validation), len(group) - n_train - 1)
        split["train"].extend(group[:n_train].tolist())
        split["validation"].extend(group[n_train:n_train + n_validation].tolist())
        split["test"].extend(group[n_train + n_validation:].tolist())
    return split


def smooth_trip(frame: pd.DataFrame, columns: list[str], window: int) -> pd.DataFrame:
    """Apply a trailing moving average within one trip; missing values remain missing."""
    values = frame[columns].apply(pd.to_numeric, errors="coerce")
    smoothed = values.rolling(window=window, min_periods=window).mean()
    # Never convert a source missing value into a measured or smoothed value.
    smoothed = smoothed.mask(values.isna())
    return smoothed


def prepare_dataset(config_path: str | Path = "config.yaml", *, data_dir: str | Path | None = None,
                    output_dir: str | Path | None = None) -> dict:
    config = load_config(config_path)
    cfg = config["data"]
    root = Path(data_dir or cfg.get("data_dir", "data/extracted"))
    out = Path(output_dir or config["project"].get("processed_dir", "data/processed"))
    xcols, ycols = cfg["inputs"], cfg["targets"]
    required = xcols + ycols
    eligible, excluded, frames = [], {}, {}
    for path in list_trip_files(root):
        frame = read_trip(path, encoding=cfg.get("encoding", "cp1252"))
        missing_columns = validate_features(frame, xcols, ycols, trip_id=path.stem)
        if missing_columns:
            excluded[path.stem] = {"reason": "missing required columns", "columns": missing_columns}
            continue
        frames[path.stem] = frame
        eligible.append(path.stem)
    if not eligible:
        raise ValueError("No trips contain all ten input and four target columns")

    split = split_trip_ids(eligible, int(config["project"]["seed"]), float(cfg["train_fraction"]),
                           float(cfg["validation_fraction"]))
    factor = int(cfg.get("downsample_factor", 1))
    smoothed = {
        tid: smooth_trip(frame, required, int(cfg["moving_average_window"])).iloc[::factor].reset_index(drop=True)
        for tid, frame in frames.items()
    }
    train_values = pd.concat([smoothed[tid] for tid in split["train"]], ignore_index=True)
    x_scaler, y_scaler = MinMaxScaler((-1, 1)), MinMaxScaler((-1, 1))
    x_scaler.fit(train_values[xcols].dropna(how="all").dropna(subset=xcols))
    y_scaler.fit(train_values[ycols].dropna(how="all").dropna(subset=ycols))

    arrays_dir = out / "trips"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    for tid, values in smoothed.items():
        x = x_scaler.transform(values[xcols])
        y = y_scaler.transform(values[ycols])
        np.savez_compressed(arrays_dir / f"{tid}.npz", x=x.astype(np.float32), y=y.astype(np.float32))
    joblib.dump({"x": x_scaler, "y": y_scaler, "x_columns": xcols, "y_columns": ycols}, out / "scalers.joblib")
    (out / "splits.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    (out / "excluded_trips.json").write_text(json.dumps(excluded, indent=2), encoding="utf-8")

    length = int(cfg["sequence_length"])
    stride = int(cfg["sequence_stride"])
    counts = {}
    for name, trip_ids in split.items():
        count = 0
        for tid in trip_ids:
            z = np.load(arrays_dir / f"{tid}.npz")
            finite = np.isfinite(z["x"]).all(axis=1) & np.isfinite(z["y"]).all(axis=1)
            for start in range(0, max(0, len(finite) - length + 1), stride):
                count += int(finite[start:start + length].all())
        counts[name] = count
    summary = {
        "eligible_trip_count": len(eligible), "eligible_trip_ids": sorted(eligible),
        "excluded_trip_count": len(excluded), "excluded_trips": excluded,
        "split_trip_ids": split, "valid_window_counts": counts,
        "total_valid_windows": sum(counts.values()), "paper_reported_windows": 87536,
        "window_count_difference": sum(counts.values()) - 87536,
        "sequence_length": length, "stride": stride,
        "moving_average_window_raw_samples": int(cfg["moving_average_window"]),
        "downsample_factor": factor,
        "effective_sequence_step_seconds": float(cfg["sample_interval_seconds"]) * factor,
        "note": "A 10-point moving average is followed by decimation to 1-second steps, consistent with Figure 1 axis labels. Windows with any missing required input or target are excluded; no missing value is imputed.",
    }
    (out / "preprocessing_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    print(json.dumps(prepare_dataset(args.config, data_dir=args.data_dir, output_dir=args.output_dir), indent=2))


if __name__ == "__main__":
    main()


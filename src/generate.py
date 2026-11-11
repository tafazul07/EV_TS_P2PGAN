"""Generate clearly labelled synthetic target signals from conditioning CSV data."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from .data_loader import read_trip, validate_features
from .generator import Generator
from .preprocessing import smooth_trip
from .utils import choose_device, load_config


@torch.no_grad()
def generate(input_csv: str | Path, checkpoint="checkpoints/best.pt", config_path="config.yaml",
             sequences=1, output="results/generated_data/synthetic.csv"):
    config = load_config(config_path)
    dc, pc = config["data"], config["project"]
    frame = read_trip(input_csv, encoding=dc.get("encoding", "cp1252"))
    absent = [c for c in dc["inputs"] if c not in frame]
    if absent:
        raise ValueError(f"Input trip misses conditioning columns {absent}; no values are fabricated")
    smooth = smooth_trip(frame, dc["inputs"], int(dc["moving_average_window"]))
    factor = int(dc.get("downsample_factor", 1))
    smooth = smooth.iloc[::factor].reset_index(drop=True)
    scalers = joblib.load(Path(pc["processed_dir"]) / "scalers.joblib")
    x = scalers["x"].transform(smooth[dc["inputs"]])
    length = int(dc["sequence_length"])
    starts = [i for i in range(max(0, len(x) - length + 1)) if np.isfinite(x[i:i + length]).all()]
    if not starts:
        raise ValueError("No complete conditioning windows in input trip")
    if sequences < 1 or sequences > len(starts):
        raise ValueError(f"Requested {sequences} sequences; input has {len(starts)} valid windows")
    starts = starts[:sequences]
    device = choose_device(config["training"].get("device", "auto"))
    model = Generator(dropout=config["model"]["generator"]["dropout"]).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["generator"]); model.eval()
    out_rows = []
    for seq_no, start in enumerate(starts, 1):
        condition = torch.from_numpy(x[start:start + length].T[None].astype(np.float32)).to(device)
        y = model(condition).cpu().numpy()[0].T
        y = scalers["y"].inverse_transform(y)
        for offset, values in enumerate(y):
            row = {name: float(value) for name, value in zip(dc["targets"], values)}
            row.update({"source_trip": Path(input_csv).stem, "sequence_number": seq_no,
                        "source_row_index": (start + offset) * factor, "record_type": "synthetic"})
            out_rows.append(row)
    dest = Path(output); dest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(dest, index=False)
    return dest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--sequences", type=int, default=1)
    parser.add_argument("--output", default="results/generated_data/synthetic.csv")
    args = parser.parse_args()
    print(generate(args.input_csv, args.checkpoint, args.config, args.sequences, args.output))


if __name__ == "__main__":
    main()


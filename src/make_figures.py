"""Create inspection figures from measured data and the documented method."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from .data_loader import list_trip_files, read_trip
from .preprocessing import smooth_trip
from .utils import load_config


def _box(ax, x, y, w, h, text, face="#E8F0F8"):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                           linewidth=1.4, edgecolor="#315A7D", facecolor=face)
    ax.add_patch(patch)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=10, wrap=True)


def create_figures(config_path="config.yaml"):
    cfg = load_config(config_path)
    out = Path(cfg["project"]["results_dir"]) / "plots"
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in list_trip_files(cfg["project"]["data_dir"]):
        rows.append((path.stem, len(pd.read_csv(path, sep=";", encoding="cp1252"))))
    frame = pd.DataFrame(rows, columns=["trip", "records"])
    colors = ["#4C78A8" if x.startswith("TripA") else "#F58518" for x in frame.trip]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(frame.trip, frame.records, color=colors)
    ax.set(title="Raw records per trip", ylabel="Rows at 100 ms sampling")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.grid(axis="y", alpha=.2); fig.tight_layout(); fig.savefig(out / "trip_lengths.png", dpi=170); plt.close(fig)

    b01 = read_trip("data/extracted/TripB01.csv")
    cfg_data = cfg["data"]
    cols = cfg_data["targets"]
    smoothed = smooth_trip(b01, cols, cfg_data["moving_average_window"]).iloc[::cfg_data["downsample_factor"]]
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    labels = ["SOC (%)", "Battery voltage (V)", "Motor torque (Nm)", "Longitudinal acceleration (m/s²)"]
    t = pd.to_numeric(b01["Time [s]"], errors="coerce").iloc[::cfg_data["downsample_factor"]].to_numpy()[:len(smoothed)]
    for ax, column, label in zip(axes, cols, labels):
        ax.plot(t, smoothed[column].to_numpy(), color="#2C6E9B", linewidth=.9)
        ax.set_ylabel(label); ax.grid(alpha=.2)
    axes[-1].set_xlabel("Trip time (s)")
    fig.suptitle("Measured TripB01 target signals after project smoothing")
    fig.tight_layout(); fig.savefig(out / "measured_trip_b01_targets.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 3.5)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    stages = [
        (0.02, "10 condition\nchannels\n10 × 256"),
        (0.22, "Encoder\n64 → 1024\n4 downsampling layers"),
        (0.42, "9 residual\nblocks\n1024 channels"),
        (0.62, "Decoder\n1024 → 64\n4 upsampling layers"),
        (0.82, "4 generated\ntargets\n4 × 256"),
    ]
    for x, text in stages: _box(ax, x, .34, .16, .34, text)
    for x in (.185, .385, .585, .785): ax.annotate("", (x + .03, .51), (x, .51), arrowprops={"arrowstyle":"->", "lw":1.5, "color":"#315A7D"})
    ax.text(.5, .12, "The discriminator receives the 10 conditions concatenated with either real or generated targets.\nD1 evaluates full resolution; D2 evaluates a pooled half-resolution sequence.", ha="center", va="center", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "ts_p2pgan_architecture.png", dpi=180, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 2.6)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    labels = ["Raw trips\n70 files", "Feature audit\n50 complete trips", "10-point average\nwithin trip", "Keep every 10th\n1 s/step", "Trip-level\nsplit", "Valid 256-step\nwindows"]
    xs = [.01, .18, .35, .52, .69, .84]
    ws = [.14, .15, .15, .15, .13, .15]
    for x, w, text in zip(xs, ws, labels): _box(ax, x, .35, w, .34, text, "#F1F5F8")
    for i in range(5): ax.annotate("", (xs[i+1]-.005, .52), (xs[i]+ws[i]+.005, .52), arrowprops={"arrowstyle":"->", "lw":1.4, "color":"#315A7D"})
    fig.tight_layout(); fig.savefig(out / "preprocessing_flow.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    return sorted(p.name for p in out.glob("*.png"))


if __name__ == "__main__":
    print("\n".join(create_figures()))


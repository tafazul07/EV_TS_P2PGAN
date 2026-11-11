"""Evaluate held-out trip windows and save metrics and real/synthetic plots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from .dataset import TripWindowDataset
from .generator import Generator
from .utils import choose_device, load_config, set_seed


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Exact DTW with O(n) memory; sequences should be modest in length."""
    previous = np.full(len(b) + 1, np.inf, dtype=np.float64)
    previous[0] = 0.0
    for av in a:
        current = np.full(len(b) + 1, np.inf, dtype=np.float64)
        for j, bv in enumerate(b, 1):
            current[j] = abs(float(av) - float(bv)) + min(current[j - 1], previous[j], previous[j - 1])
        previous = current
    return float(previous[-1] / max(len(a), len(b)))


@torch.no_grad()
def evaluate(config_path="config.yaml", checkpoint="checkpoints/best.pt", max_windows=2000):
    config = load_config(config_path)
    set_seed(int(config["project"]["seed"]))
    device = choose_device(config["training"].get("device", "auto"))
    model_cfg, data_cfg, project = config["model"], config["data"], config["project"]
    model = Generator(residual_blocks=model_cfg["generator"]["residual_blocks"], dropout=model_cfg["generator"]["dropout"]).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["generator"])
    model.eval()
    ds = TripWindowDataset(project["processed_dir"], "test", data_cfg["sequence_length"], data_cfg["sequence_stride"], max_windows)
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    arrays = {"real": [], "synthetic": []}
    for x, y in loader:
        arrays["real"].append(y.numpy())
        arrays["synthetic"].append(model(x.to(device)).cpu().numpy())
    if not arrays["real"]:
        raise ValueError("No held-out windows; preprocess the data and inspect missing targets")
    real_n = np.concatenate(arrays["real"])
    fake_n = np.concatenate(arrays["synthetic"])
    scalers = joblib.load(Path(project["processed_dir"]) / "scalers.joblib")
    n, c, length = real_n.shape
    real = scalers["y"].inverse_transform(real_n.transpose(0, 2, 1).reshape(-1, c)).reshape(n, length, c)
    fake = scalers["y"].inverse_transform(fake_n.transpose(0, 2, 1).reshape(-1, c)).reshape(n, length, c)
    names = [x.replace(" [", " (").replace("]", ")") for x in scalers["y_columns"]]
    metrics = []
    for i, name in enumerate(names):
        delta = real[:, :, i] - fake[:, :, i]
        rmse = float(np.sqrt(np.mean(delta ** 2)))
        mae = float(np.mean(np.abs(delta)))
        # Cap expensive DTW to a reproducible first subset and 256 points per window.
        dtw = float(np.mean([dtw_distance(real[j, ::4, i], fake[j, ::4, i]) for j in range(min(n, 100))]))
        metrics.append({"feature": name, "rmse_physical_units": rmse, "mae_physical_units": mae,
                        "dtw_mean_abs_per_sample": dtw, "paired_test_windows": n,
                        "dtw_windows": min(n, 100), "dtw_downsample_factor": 4})
    out = Path(project["results_dir"])
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    (out / "metrics" / "heldout_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    summary = {}
    for i, name in enumerate(names):
        summary[name] = {
            "measured_mean": float(real[:, :, i].mean()), "measured_std": float(real[:, :, i].std()),
            "measured_min": float(real[:, :, i].min()), "measured_max": float(real[:, :, i].max()),
            "synthetic_mean": float(fake[:, :, i].mean()), "synthetic_std": float(fake[:, :, i].std()),
            "synthetic_min": float(fake[:, :, i].min()), "synthetic_max": float(fake[:, :, i].max()),
        }
    (out / "metrics" / "heldout_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot_results(real, fake, names, out / "plots", int(project["seed"]))
    return metrics


def _plot_results(real, fake, names, plot_dir: Path, seed: int):
    # One representative held-out paired sequence per output feature.
    fig, axes = plt.subplots(len(names), 1, figsize=(12, 10), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(real[0, :, i], label="Measured held-out", linewidth=1)
        ax.plot(fake[0, :, i], label="Synthetic conditioned output", linewidth=1)
        ax.set_ylabel(names[i]); ax.legend(loc="best")
    axes[-1].set_xlabel("Sample index (100 ms per step)")
    fig.suptitle("Paired real and synthetic outputs on a held-out trip")
    fig.tight_layout(); fig.savefig(plot_dir / "real_vs_synthetic.png", dpi=160); plt.close(fig)

    flat_real, flat_fake = real.reshape(len(real), -1), fake.reshape(len(fake), -1)
    rng = np.random.default_rng(seed)
    count = min(1000, len(real))
    ids = rng.choice(len(real), count, replace=False)
    combined = np.concatenate((flat_real[ids], flat_fake[ids]), axis=0)
    labels = np.array(["real"] * count + ["synthetic"] * count)
    pca = PCA(n_components=2, random_state=seed).fit_transform(combined)
    fig, ax = plt.subplots(figsize=(8, 6))
    for label in ("real", "synthetic"):
        m = labels == label; ax.scatter(pca[m, 0], pca[m, 1], s=10, alpha=.55, label=label)
    ax.set(title="PCA: held-out real and synthetic sequences", xlabel="PC1", ylabel="PC2"); ax.legend()
    fig.tight_layout(); fig.savefig(plot_dir / "pca.png", dpi=160); plt.close(fig)
    perplexity = min(30, max(2, len(combined) // 10))
    embedded = TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto", perplexity=perplexity).fit_transform(combined)
    fig, ax = plt.subplots(figsize=(8, 6))
    for label in ("real", "synthetic"):
        m = labels == label; ax.scatter(embedded[m, 0], embedded[m, 1], s=10, alpha=.55, label=label)
    ax.set(title="t-SNE: held-out real and synthetic sequences", xlabel="Dimension 1", ylabel="Dimension 2"); ax.legend()
    fig.tight_layout(); fig.savefig(plot_dir / "tsne.png", dpi=160); plt.close(fig)

    rflat = real.reshape(-1, len(names)); fflat = fake.reshape(-1, len(names))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for i, ax in enumerate(axes.flat):
        ax.hist(rflat[:, i], bins=40, alpha=.55, density=True, label="Measured")
        ax.hist(fflat[:, i], bins=40, alpha=.55, density=True, label="Synthetic")
        ax.set_title(names[i]); ax.legend()
    fig.suptitle("Feature distributions: held-out windows")
    fig.tight_layout(); fig.savefig(plot_dir / "feature_distributions.png", dpi=160); plt.close(fig)

    corr_real = np.corrcoef(real.reshape(-1, len(names)), rowvar=False)
    corr_fake = np.corrcoef(fake.reshape(-1, len(names)), rowvar=False)
    pd.DataFrame(corr_real, index=names, columns=names).to_csv(plot_dir.parent / "metrics" / "real_correlation.csv")
    pd.DataFrame(corr_fake, index=names, columns=names).to_csv(plot_dir.parent / "metrics" / "synthetic_correlation.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, matrix, title in zip(axes, (corr_real, corr_fake), ("Measured", "Synthetic")):
        image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
        ax.set_yticks(range(len(names)), names); ax.set_title(f"{title} target correlation")
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Pearson correlation", shrink=.8)
    fig.tight_layout(); fig.savefig(plot_dir / "correlation_matrices.png", dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--max-windows", type=int, default=2000)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.config, args.checkpoint, args.max_windows), indent=2))


if __name__ == "__main__":
    main()


"""Train TS-p2pGAN. Run with ``python -m src.train --smoke`` for a short check."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import TripWindowDataset
from .discriminator import MultiScaleDiscriminator
from .generator import Generator
from .losses import MultiScaleGANLoss, generator_loss
from .utils import choose_device, load_config, save_checkpoint, set_seed


def _finite(*values: torch.Tensor) -> bool:
    return all(torch.isfinite(v).all().item() for v in values)


def _set_lr(optimizers, base_lr: float, epoch: int, epochs: int, decay_epochs: int) -> float:
    if decay_epochs <= 0:
        return base_lr
    start = epochs - decay_epochs
    factor = 1.0 if epoch < start else max(0.0, (epochs - epoch - 1) / max(1, decay_epochs))
    lr = base_lr * factor
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["lr"] = lr
    return lr


def _epoch(generator, discriminator, loader, device, lambda_l1, optimizer_g=None, optimizer_d=None):
    training = optimizer_g is not None
    generator.train(training)
    discriminator.train(training)
    gan_loss = MultiScaleGANLoss()
    totals = {"g": 0.0, "g_gan": 0.0, "g_l1": 0.0, "d": 0.0}
    batches = 0
    for x, y in tqdm(loader, leave=False, desc="train" if training else "validation"):
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(training):
            fake = generator(x)
            if not _finite(x, y, fake):
                raise FloatingPointError("NaN or infinite tensor detected in GAN batch")
            if training:
                optimizer_d.zero_grad(set_to_none=True)
            real_pred = discriminator(x, y)
            fake_pred = discriminator(x, fake.detach())
            d_loss = gan_loss(real_pred, True) + gan_loss(fake_pred, False)
            if training:
                d_loss.backward()
                optimizer_d.step()
                optimizer_g.zero_grad(set_to_none=True)
            fake_pred_g = discriminator(x, fake)
            g_loss, g_gan, g_l1 = generator_loss(fake_pred_g, fake, y, lambda_l1, gan_loss)
            if training:
                g_loss.backward()
                optimizer_g.step()
            if not _finite(d_loss.detach(), g_loss.detach()):
                raise FloatingPointError("NaN or infinite GAN loss detected")
        totals["g"] += float(g_loss.detach())
        totals["g_gan"] += float(g_gan.detach())
        totals["g_l1"] += float(g_l1.detach())
        totals["d"] += float(d_loss.detach())
        batches += 1
    if batches == 0:
        raise ValueError("No valid sequence batches found; run preprocessing and check missing data")
    return {key: value / batches for key, value in totals.items()}


def train(config_path="config.yaml", *, smoke=False, resume=None, epochs_override=None):
    config = load_config(config_path)
    set_seed(int(config["project"]["seed"]))
    tc = dict(config["training"])
    pc = config["project"]
    dc = config["data"]
    if smoke:
        tc.update(tc["smoke"])
    epochs = int(epochs_override or tc["epochs"])
    batch_size = int(tc["batch_size"])
    processed = pc["processed_dir"]
    limit = tc.get("max_sequences") if smoke else None
    train_ds = TripWindowDataset(processed, "train", dc["sequence_length"], dc["sequence_stride"], limit)
    val_ds = TripWindowDataset(processed, "validation", dc["sequence_length"], dc["sequence_stride"],
                               max_sequences=limit if smoke else None)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=int(tc["num_workers"]))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=int(tc["num_workers"]))
    device = choose_device(tc.get("device", "auto"))
    mc = config["model"]
    generator = Generator(residual_blocks=mc["generator"]["residual_blocks"], dropout=mc["generator"]["dropout"]).to(device)
    discriminator = MultiScaleDiscriminator().to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=tc["learning_rate"], betas=(tc["beta1"], tc["beta2"]))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=tc["learning_rate"], betas=(tc["beta1"], tc["beta2"]))
    start_epoch, best_val = 0, math.inf
    if resume:
        state = torch.load(resume, map_location=device, weights_only=False)
        generator.load_state_dict(state["generator"])
        discriminator.load_state_dict(state["discriminator"])
        opt_g.load_state_dict(state["optimizer_g"])
        opt_d.load_state_dict(state["optimizer_d"])
        start_epoch = int(state["epoch"]) + 1
        best_val = float(state.get("best_val_loss", math.inf))

    out_dir = Path(pc["results_dir"]) / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(pc["checkpoints_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / ("smoke_training.csv" if smoke else "training.csv")
    write_header = not log_path.exists() or not resume
    with log_path.open("w" if write_header else "a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "lr", "train_g", "train_gan", "train_l1", "train_d", "val_g", "val_gan", "val_l1", "val_d"])
        if write_header:
            writer.writeheader()
        for epoch in range(start_epoch, epochs):
            effective_decay = min(int(tc["decay_epochs"]), max(0, epochs // 4))
            lr = _set_lr((opt_g, opt_d), tc["learning_rate"], epoch, epochs, effective_decay)
            train_m = _epoch(generator, discriminator, train_loader, device, mc["lambda_l1"], opt_g, opt_d)
            val_m = _epoch(generator, discriminator, val_loader, device, mc["lambda_l1"])
            writer.writerow({"epoch": epoch + 1, "lr": lr, "train_g": train_m["g"], "train_gan": train_m["g_gan"],
                             "train_l1": train_m["g_l1"], "train_d": train_m["d"], "val_g": val_m["g"],
                             "val_gan": val_m["g_gan"], "val_l1": val_m["g_l1"], "val_d": val_m["d"]})
            stream.flush()
            save_checkpoint(checkpoint_dir / "last.pt", epoch=epoch, generator=generator,
                            discriminator=discriminator, optimizer_g=opt_g, optimizer_d=opt_d,
                            best_val_loss=best_val, config=config)
            if val_m["g"] < best_val:
                best_val = val_m["g"]
                save_checkpoint(checkpoint_dir / "best.pt", epoch=epoch, generator=generator,
                                discriminator=discriminator, optimizer_g=opt_g, optimizer_d=opt_d,
                                best_val_loss=best_val, config=config)
            print(f"epoch {epoch+1}/{epochs} lr={lr:.6g} G={train_m['g']:.4f} D={train_m['d']:.4f} val_G={val_m['g']:.4f} device={device}")
    return {"epochs": epochs, "device": str(device), "train_sequences": len(train_ds), "validation_sequences": len(val_ds), "best_validation_loss": best_val}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    print(train(args.config, smoke=args.smoke, resume=args.resume, epochs_override=args.epochs))


if __name__ == "__main__":
    main()


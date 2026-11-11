"""Conditional GAN and reconstruction losses."""
from __future__ import annotations

import torch
from torch import nn


class MultiScaleGANLoss(nn.Module):
    """Binary cross-entropy over both discriminator scales."""
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, predictions: tuple[torch.Tensor, torch.Tensor], real: bool) -> torch.Tensor:
        label = 1.0 if real else 0.0
        return sum(self.bce(p, torch.full_like(p, label)) for p in predictions)


def generator_loss(fake_predictions, generated: torch.Tensor, target: torch.Tensor,
                   lambda_l1: float, gan_loss: MultiScaleGANLoss):
    adversarial = gan_loss(fake_predictions, True)
    reconstruction = nn.functional.l1_loss(generated, target)
    return adversarial + lambda_l1 * reconstruction, adversarial, reconstruction


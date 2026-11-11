"""Two-scale conditional discriminator (Table 3)."""
from __future__ import annotations

import torch
from torch import nn


def _branch(in_channels: int, first_length: int) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Conv1d(in_channels, 64, 4, stride=2, padding=2), nn.LeakyReLU(0.2, True)]
    channels = 64
    for next_channels in (128, 256):
        layers += [nn.Conv1d(channels, next_channels, 4, stride=2, padding=2, bias=False),
                   nn.BatchNorm1d(next_channels), nn.LeakyReLU(0.2, True)]
        channels = next_channels
    layers += [nn.Conv1d(channels, 512, 4, stride=1, padding=2, bias=False),
               nn.BatchNorm1d(512), nn.LeakyReLU(0.2, True),
               nn.Conv1d(512, 1, 4, stride=1, padding=2), nn.Sigmoid()]
    return nn.Sequential(*layers)


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, condition_channels: int = 10, target_channels: int = 4):
        super().__init__()
        self.input_channels = condition_channels + target_channels
        self.fine = _branch(self.input_channels, 256)
        self.pool = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
        self.coarse = _branch(self.input_channels, 128)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        pair = x if y is None else torch.cat((x, y), dim=1)
        if pair.shape[1] != self.input_channels:
            raise ValueError(f"Discriminator expects {self.input_channels} channels, got {pair.shape[1]}")
        return self.fine(pair), self.coarse(self.pool(pair))


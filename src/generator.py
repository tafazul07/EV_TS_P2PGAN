"""1D TS-p2pGAN conditional generator (Table 2)."""
from __future__ import annotations

import torch
from torch import nn


class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int = 1024, dropout: float = 0.5):
        super().__init__()
        layers: list[nn.Module] = [
            nn.ReflectionPad1d(1), nn.Conv1d(channels, channels, 3, padding=0, bias=False),
            nn.BatchNorm1d(channels), nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [
            nn.ReflectionPad1d(1), nn.Conv1d(channels, channels, 3, padding=0, bias=False),
            nn.BatchNorm1d(channels), nn.ReLU(inplace=True),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class Generator(nn.Module):
    """Map [batch,10,256] conditions to [batch,4,256] target channels."""
    def __init__(self, in_channels: int = 10, out_channels: int = 4,
                 residual_blocks: int = 9, dropout: float = 0.5):
        super().__init__()
        if residual_blocks != 9:
            raise ValueError("The paper specifies exactly nine residual blocks")
        enc: list[nn.Module] = [
            nn.ReflectionPad1d(3), nn.Conv1d(in_channels, 64, 7, padding=0, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
        ]
        channels = 64
        for next_channels in (128, 256, 512, 1024):
            enc += [nn.Conv1d(channels, next_channels, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm1d(next_channels), nn.ReLU(inplace=True)]
            channels = next_channels
        self.encoder = nn.Sequential(*enc)
        self.residual = nn.Sequential(*(ResidualBlock1D(1024, dropout) for _ in range(9)))
        dec: list[nn.Module] = []
        for next_channels in (512, 256, 128, 64):
            # output_padding=1 is needed to obtain the paper's stated exact doubling.
            dec += [nn.ConvTranspose1d(channels, next_channels, 3, stride=2, padding=1,
                                       output_padding=1, bias=False),
                    nn.BatchNorm1d(next_channels), nn.ReLU(inplace=True)]
            channels = next_channels
        dec += [nn.ReflectionPad1d(3), nn.Conv1d(channels, out_channels, 7, padding=0), nn.Tanh()]
        self.decoder = nn.Sequential(*dec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.residual(self.encoder(x)))


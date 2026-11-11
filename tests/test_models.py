import pytest
torch = pytest.importorskip("torch")

from src.discriminator import MultiScaleDiscriminator
from src.generator import Generator, ResidualBlock1D


def test_generator_shapes_and_finite_values():
    model = Generator().eval()
    x = torch.randn(1, 10, 256)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 4, 256)
    assert torch.isfinite(y).all()
    assert y.min() >= -1.0 and y.max() <= 1.0


def test_residual_dimensions_and_discriminator_scales():
    block = ResidualBlock1D().eval()
    x = torch.randn(2, 1024, 16)
    assert block(x).shape == x.shape
    disc = MultiScaleDiscriminator().eval()
    with torch.no_grad():
        fine, coarse = disc(torch.randn(2, 10, 256), torch.randn(2, 4, 256))
    assert fine.shape == (2, 1, 35)
    assert coarse.shape == (2, 1, 19)
    assert torch.isfinite(fine).all() and torch.isfinite(coarse).all()


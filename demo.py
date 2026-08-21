"""Smoke test and demo for SemanticTransferUNet."""

import torch

from models import SemanticTransferUNet
from utils import describe_device, resolve_device


def main() -> None:
    device = resolve_device("auto")
    print(f"Device: {describe_device(device)}")

    model = SemanticTransferUNet(
        in_channels=3,
        base_channels=32,
        context_dim=256,
        num_heads=8,
    ).to(device)

    batch = 1
    image = torch.randn(batch, 3, 1024, 1024, device=device)
    ref = torch.randn(batch, 3, 256, 256, device=device)

    with torch.no_grad():
        out = model.transfer(image, ref)

    assert out.shape == image.shape, f"Expected {image.shape}, got {out.shape}"

    params = sum(p.numel() for p in model.parameters())
    print(f"Output shape: {tuple(out.shape)}")
    print(f"Parameters:   {params:,}")
    print("Forward pass OK — model performs residual semantic transfer.")


if __name__ == "__main__":
    main()

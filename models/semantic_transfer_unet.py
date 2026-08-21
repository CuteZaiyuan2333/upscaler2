"""
Semantic Transfer U-Net.

Main path:  1024x1024 input  -> 1024x1024 output (residual semantic transfer)
Context:    256x256 reference image -> cross-attention K/V tokens

The network learns a residual delta; output = input + delta.
It does not generate images from scratch — it transfers semantics from the
reference image onto the input image structure.
"""

import torch
import torch.nn as nn

from .reference_encoder import ReferenceImageEncoder
from .unet_blocks import BottleneckBlock, DownBlock, ResidualConvBlock, UpBlock


class SemanticTransferUNet(nn.Module):
    """
    U-Net with image-based cross-attention for semantic transfer.

    Args:
        in_channels: input image channels (default 3).
        base_channels: base feature width (default 64).
        context_dim: cross-attention context dimension (default 512).
        num_heads: attention heads per cross-attention block.
        residual_scale: scales the learned delta before adding to input.
    """

    INPUT_SIZE = 1024
    REF_SIZE = 256

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        context_dim: int = 512,
        num_heads: int = 8,
        residual_scale: float = 1.0,
    ):
        super().__init__()
        self.residual_scale = residual_scale
        self.context_dim = context_dim

        c = base_channels
        chs = [c, c * 2, c * 4, c * 8, c * 8, c * 8]

        self.ref_encoder = ReferenceImageEncoder(
            in_channels=in_channels,
            base_channels=c,
            context_dim=context_dim,
        )

        self.stem = ResidualConvBlock(in_channels, chs[0])

        self.down1 = DownBlock(chs[0], chs[1])
        self.down2 = DownBlock(chs[1], chs[2])
        self.down3 = DownBlock(chs[2], chs[3])
        self.down4 = DownBlock(chs[3], chs[4])
        self.down5 = DownBlock(chs[4], chs[5])

        self.bottleneck = BottleneckBlock(chs[5], context_dim, num_heads=num_heads)

        self.up5 = UpBlock(chs[5], chs[5], chs[4], context_dim, num_heads, use_cross_attn=True)
        self.up4 = UpBlock(chs[4], chs[4], chs[3], context_dim, num_heads, use_cross_attn=True)
        self.up3 = UpBlock(chs[3], chs[3], chs[2], context_dim, num_heads, use_cross_attn=True)
        # 512² and 1024² cross-attn need multi-GB attention maps; semantics are
        # injected at coarser scales, fine detail comes from skip connections.
        self.up2 = UpBlock(chs[2], chs[2], chs[1], context_dim, num_heads, use_cross_attn=False)
        self.up1 = UpBlock(chs[1], chs[1], chs[0], context_dim, num_heads, use_cross_attn=False)

        self.out_conv = nn.Sequential(
            nn.Conv2d(chs[0], chs[0], 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(chs[0], in_channels, 3, padding=1),
        )

        self._init_output_zero()

    def _init_output_zero(self) -> None:
        """Initialize output conv to zero so the model starts as identity."""
        nn.init.zeros_(self.out_conv[-1].weight)
        nn.init.zeros_(self.out_conv[-1].bias)

    def encode_reference(self, ref_image: torch.Tensor) -> torch.Tensor:
        """Encode 256x256 reference image to cross-attention context tokens."""
        if ref_image.shape[-2:] != (self.REF_SIZE, self.REF_SIZE):
            raise ValueError(
                f"Reference image must be {self.REF_SIZE}x{self.REF_SIZE}, "
                f"got {ref_image.shape[-2]}x{ref_image.shape[-1]}"
            )
        return self.ref_encoder(ref_image)

    def forward_features(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        x = self.stem(x)

        x, s1 = self.down1(x)
        x, s2 = self.down2(x)
        x, s3 = self.down3(x)
        x, s4 = self.down4(x)
        x, s5 = self.down5(x)

        x = self.bottleneck(x, context)

        x = self.up5(x, s5, context)
        x = self.up4(x, s4, context)
        x = self.up3(x, s3, context)
        x = self.up2(x, s2, context)
        x = self.up1(x, s1, context)

        return self.out_conv(x)

    def forward(
        self,
        image: torch.Tensor,
        ref_image: torch.Tensor,
    ) -> torch.Tensor:
        """
        Transfer semantics from ref_image onto image.

        Args:
            image: [B, 3, 1024, 1024] input image to modify.
            ref_image: [B, 3, 256, 256] reference carrying target semantics.

        Returns:
            [B, 3, 1024, 1024] input + learned semantic residual.
        """
        if image.shape[-2:] != (self.INPUT_SIZE, self.INPUT_SIZE):
            raise ValueError(
                f"Input image must be {self.INPUT_SIZE}x{self.INPUT_SIZE}, "
                f"got {image.shape[-2]}x{image.shape[-1]}"
            )

        context = self.encode_reference(ref_image)
        delta = self.forward_features(image, context)
        return image + self.residual_scale * delta

    @torch.no_grad()
    def transfer(
        self,
        image: torch.Tensor,
        ref_image: torch.Tensor,
    ) -> torch.Tensor:
        """Inference helper (same as forward, explicit name)."""
        return self.forward(image, ref_image)

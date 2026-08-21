import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ReferenceImageEncoder(nn.Module):
    """
    Encodes a 256x256 reference image into a token sequence for cross-attention.

    256 -> 128 -> 64 -> 32 -> 16 spatial resolution, yielding 16x16 = 256 tokens.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        context_dim: int = 512,
    ):
        super().__init__()
        self.context_dim = context_dim

        ch = base_channels
        self.stem = ConvBlock(in_channels, ch)
        self.down1 = nn.Sequential(nn.Conv2d(ch, ch * 2, 4, stride=2, padding=1), ConvBlock(ch * 2, ch * 2))
        self.down2 = nn.Sequential(nn.Conv2d(ch * 2, ch * 4, 4, stride=2, padding=1), ConvBlock(ch * 4, ch * 4))
        self.down3 = nn.Sequential(nn.Conv2d(ch * 4, ch * 8, 4, stride=2, padding=1), ConvBlock(ch * 8, ch * 8))
        self.down4 = nn.Sequential(nn.Conv2d(ch * 8, ch * 8, 4, stride=2, padding=1), ConvBlock(ch * 8, ch * 8))

        self.proj = nn.Conv2d(ch * 8, context_dim, 1)

    def forward(self, ref_image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ref_image: [B, 3, 256, 256] reference image carrying target semantics.
        Returns:
            context: [B, 256, context_dim] token sequence for cross-attention.
        """
        x = self.stem(ref_image)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.proj(x)

        b, c, h, w = x.shape
        return x.flatten(2).transpose(1, 2)

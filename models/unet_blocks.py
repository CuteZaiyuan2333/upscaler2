import torch
import torch.nn as nn

from .cross_attention import CrossAttentionBlock


class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(32, out_ch), out_ch),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x) + self.skip(x))


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = ResidualConvBlock(in_ch, out_ch)
        self.down = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.block(x)
        skip = x
        return self.down(x), skip


class UpBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        context_dim: int,
        num_heads: int = 8,
        use_cross_attn: bool = True,
    ):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.block = ResidualConvBlock(out_ch + skip_ch, out_ch)
        self.use_cross_attn = use_cross_attn
        if use_cross_attn:
            self.cross_attn = CrossAttentionBlock(out_ch, context_dim, num_heads=num_heads)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.block(torch.cat([x, skip], dim=1))

        if self.use_cross_attn:
            b, c, h, w = x.shape
            tokens = x.flatten(2).transpose(1, 2)
            tokens = self.cross_attn(tokens, context)
            x = tokens.transpose(1, 2).reshape(b, c, h, w)

        return x


class BottleneckBlock(nn.Module):
    def __init__(self, channels: int, context_dim: int, num_heads: int = 8):
        super().__init__()
        self.conv = ResidualConvBlock(channels, channels)
        self.cross_attn = CrossAttentionBlock(channels, context_dim, num_heads=num_heads)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.cross_attn(tokens, context)
        return tokens.transpose(1, 2).reshape(b, c, h, w)

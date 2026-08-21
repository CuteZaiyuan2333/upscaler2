import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageCrossAttention(nn.Module):
    """Cross-attention where K/V come from a reference image feature sequence."""

    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        num_heads: int = 8,
        head_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = head_dim * num_heads if head_dim is not None else query_dim
        if head_dim is None:
            head_dim = inner_dim // num_heads

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, N, query_dim] spatial tokens from main U-Net.
            context: [B, M, context_dim] tokens from reference image encoder.
            context_mask: optional [B, M] bool mask (True = keep).
        """
        b, n, _ = hidden_states.shape
        m = context.shape[1]

        q = self.to_q(hidden_states)
        k = self.to_k(context)
        v = self.to_v(context)

        q = q.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, m, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, m, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if context_mask is not None:
            mask = context_mask[:, None, None, :]
            attn = attn.masked_fill(~mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, n, -1)

        return self.to_out(out)


class CrossAttentionBlock(nn.Module):
    """Pre-norm transformer block: LayerNorm -> CrossAttn -> residual."""

    def __init__(
        self,
        dim: int,
        context_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = ImageCrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return x + self.attn(self.norm(x), context, context_mask)

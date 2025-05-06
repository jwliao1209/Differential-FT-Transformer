import math

import torch
from torch import nn


def get_pos_emb_encoder(pos_emb_name: str) -> nn.Module:
    match pos_emb_name.lower():
        case "sin":
            return SinusoidalPEEncoder
        case "rotary":
            return RotaryPEEncoder
        case _:
            raise ValueError(f"Unknown positional embedding type: {pos_emb_name}")


class SinusoidalPEEncoder(nn.Module):
    def __init__(self, dim: int, max_scale: int = 1000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_scale = max_scale

        inv_freq = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(max_scale) / dim)
        )
        self.register_buffer('inv_freq', inv_freq)
        self.comb_linear = nn.Linear(2 * dim, dim)

    def forward(self, x: torch.Tensor, x_quantile: torch.Tensor) -> torch.Tensor:
        """
        x_quantile: Tensor, shape = (batch, num_features), values in [0, 1]
        Returns:
            sinusoidal embedding: shape = (batch, num_features, dim)
        """
        B, F = x_quantile.shape
        pos = x_quantile.unsqueeze(-1)  # (B, F, 1)
        angle = pos * self.inv_freq     # (B, F, dim // 2)
        quantile_emb = torch.zeros(B, F, self.dim, device=x_quantile.device)
        quantile_emb[..., 0::2] = torch.sin(angle)
        quantile_emb[..., 1::2] = torch.cos(angle)
        quantile_emb = quantile_emb.unsqueeze(2)

        quantile_emb = quantile_emb.expand(x.shape)
        x_cat = torch.cat([x, quantile_emb], dim=-1)
        return self.comb_linear(x_cat)


class RotaryPEEncoder(nn.Module):
    def __init__(self, dim: int, max_scale: float = 1000.0):
        super().__init__()
        assert dim % 2 == 0, "d_model must be even"
        self.dim = dim

        inv_freq = torch.exp(
            torch.arange(0, dim // 2).float() * (-math.log(max_scale) / (dim // 2))
        )  # shape: (dim // 2,)
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (..., dim)
            pos: Tensor of shape (...,) in [0, 1] range or scalar position

        Returns:
            rotated x with RoPE applied, same shape as x
        """
        # Expand pos to shape (..., 1)
        angle = (pos.unsqueeze(-1) * self.inv_freq).unsqueeze(2)  # (..., dim // 2)
        x1, x2 = x[..., ::2], x[..., 1::2]
        sin = angle.sin().expand(x1.shape)
        cos = angle.cos().expand(x1.shape)
        x_rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return x_rotated.flatten(-2)

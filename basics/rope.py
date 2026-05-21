"""Rotary Position Embeddings — §6.

You implement: RoPE1D, RoPE2D.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _apply_rotation(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply 2D rotation to paired dimensions.

    Args:
        x: (..., head_dim) with head_dim even.
        cos, sin: broadcastable to (..., head_dim // 2).
    """
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rot_even = x_even * cos - x_odd * sin
    rot_odd = x_even * sin + x_odd * cos
    return torch.stack((rot_even, rot_odd), dim=-1).flatten(-2)


class RoPE1D(nn.Module):
    """1D Rotary Position Embedding."""

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10_000.0) -> None:
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = base ** (-torch.arange(0, head_dim, 2).float() / head_dim)
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_cached[positions]
        sin = self.sin_cached[positions]
        while cos.ndim < x.ndim:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        return _apply_rotation(x, cos, sin)


class RoPE2D(nn.Module):
    """2D Rotary Position Embedding for image patches."""

    def __init__(self, head_dim: int, grid_size: int, base: float = 10_000.0) -> None:
        super().__init__()
        assert head_dim % 4 == 0, "head_dim must be divisible by 4 for 2D RoPE"
        self.head_dim = head_dim
        self.grid_size = grid_size
        self.base = base
        half_dim = head_dim // 2
        quarter_dim = head_dim // 4

        inv_freq = base ** (-torch.arange(0, half_dim, 2).float() / half_dim)
        coords = torch.arange(grid_size).float()
        freqs = torch.outer(coords, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        self.register_buffer("cos_x", cos, persistent=False)
        self.register_buffer("sin_x", sin, persistent=False)
        self.register_buffer("cos_y", cos, persistent=False)
        self.register_buffer("sin_y", sin, persistent=False)
        self.quarter_dim = quarter_dim

    def forward(
        self,
        x: torch.Tensor,
        x_coords: torch.Tensor,
        y_coords: torch.Tensor,
    ) -> torch.Tensor:
        x_half, y_half = x.split(self.head_dim // 2, dim=-1)
        cos_x = self.cos_x[x_coords]
        sin_x = self.sin_x[x_coords]
        cos_y = self.cos_y[y_coords]
        sin_y = self.sin_y[y_coords]
        while cos_x.ndim < x.ndim:
            cos_x = cos_x.unsqueeze(0)
            sin_x = sin_x.unsqueeze(0)
            cos_y = cos_y.unsqueeze(0)
            sin_y = sin_y.unsqueeze(0)
        x_rot = _apply_rotation(x_half, cos_x, sin_x)
        y_rot = _apply_rotation(y_half, cos_y, sin_y)
        return torch.cat((x_rot, y_rot), dim=-1)

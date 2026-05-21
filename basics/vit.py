"""Vision Transformer — §2.

You implement: PatchEmbeddings, ViT.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import basics
from basics.rope import RoPE1D, RoPE2D


class PatchEmbeddings(nn.Module):
    """Split an image into non-overlapping patches and project each to d_model.

    Implemented with a strided Conv2d whose kernel size and stride both equal
    `patch_size`.

    Args:
        img_size:   Input image side length (assumed square). Must be divisible
                    by patch_size.
        patch_size: Side length of each patch in pixels.
        d_model:    Output embedding dimension per patch.

    Forward:
        x: (B, 3, img_size, img_size) float tensor.
        returns: (B, num_patches, d_model) where num_patches = (img_size // patch_size) ** 2.
    """

    def __init__(self, img_size: int, patch_size: int, d_model: int) -> None:
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.conv = nn.Conv2d(
            kernel_size=patch_size, stride=patch_size, in_channels=3, out_channels=d_model
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, 3, img_size, img_size) -> (B, d_model, img_size / patch_size, img_size / patch_size)
        x = self.conv(x)
        # (B, d_model, K, K) -> (B, K * K, d_model)
        x = rearrange(x, "B d K1 K2 -> B (K1 K2) d")
        return x


class RoPEMultiHeadAttention(nn.Module):
    """Multi-head self-attention with RoPE applied to Q and K."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        pos_encoding: str,
        rope_1d: RoPE1D | None = None,
        rope_2d: RoPE2D | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.pos_encoding = pos_encoding
        self.rope_1d = rope_1d
        self.rope_2d = rope_2d
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        positions_1d: torch.Tensor | None = None,
        x_coords: torch.Tensor | None = None,
        y_coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if self.pos_encoding == "rope_1d":
            assert self.rope_1d is not None and positions_1d is not None
            q = self.rope_1d(q, positions_1d)
            k = self.rope_1d(k, positions_1d)
        elif self.pos_encoding == "rope_2d":
            assert self.rope_2d is not None and x_coords is not None and y_coords is not None
            q = self.rope_2d(q, x_coords, y_coords)
            k = self.rope_2d(k, x_coords, y_coords)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.dropout(self.out_proj(out))


class RoPEBlock(nn.Module):
    """Pre-LayerNorm Transformer block with RoPE attention."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        pos_encoding: str,
        rope_1d: RoPE1D | None = None,
        rope_2d: RoPE2D | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = RoPEMultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            pos_encoding=pos_encoding,
            rope_1d=rope_1d,
            rope_2d=rope_2d,
            dropout=dropout,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = basics.model.MLP(d_model=d_model, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        positions_1d: torch.Tensor | None = None,
        x_coords: torch.Tensor | None = None,
        y_coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.ln1(x),
            positions_1d=positions_1d,
            x_coords=x_coords,
            y_coords=y_coords,
        )
        x = x + self.mlp(self.ln2(x))
        return x


class ViT(nn.Module):
    """Vision Transformer.

    Pipeline:
      1. Patchify with `PatchEmbeddings`.
      2. Prepend a learnable [CLS] token.
      3. Add a learnable positional embedding of shape (1, num_patches+1, d_model).
      4. Pass the sequence through `num_blocks` Transformer Blocks
         (with is_decoder=False).
      5. Apply a final LayerNorm.
      6. Return only the [CLS] slice — shape (B, d_model).

    For §5 (VLM), you may want a `return_all_tokens=True` flag that returns the
    full (B, num_patches+1, d_model) sequence instead. Add it when you get there.

    Args:
        img_size, patch_size, d_model, num_heads, num_blocks, dropout
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        d_model: int,
        num_heads: int,
        num_blocks: int,
        dropout: float = 0.1,
        pos_encoding: str = "learned",
        max_img_size: int | None = None,
    ) -> None:
        super().__init__()

        # shapes
        self.img_size = img_size
        self.patch_size = patch_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.pos_encoding = pos_encoding
        self.max_img_size = max_img_size or img_size

        num_patches = (img_size // patch_size) * (img_size // patch_size)
        train_grid_side = img_size // patch_size
        max_num_patches = (self.max_img_size // patch_size) * (self.max_img_size // patch_size)
        block_size = max_num_patches + 1

        self.train_num_patches = num_patches
        self.train_grid_side = train_grid_side

        # params
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.embeddings = PatchEmbeddings(img_size, patch_size, d_model)

        if pos_encoding == "learned":
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))
            self.rope_1d = None
            self.rope_2d = None
            self.blocks = nn.ModuleList(
                [
                    basics.model.Block(
                        d_model, num_heads, block_size, is_decoder=False, dropout=dropout
                    )
                    for _ in range(num_blocks)
                ]
            )
        elif pos_encoding == "rope_1d":
            self.pos_embed = None
            head_dim = d_model // num_heads
            self.rope_1d = RoPE1D(head_dim, max_seq_len=block_size)
            self.rope_2d = None
            self.blocks = nn.ModuleList(
                [
                    RoPEBlock(
                        d_model,
                        num_heads,
                        pos_encoding="rope_1d",
                        rope_1d=self.rope_1d,
                        dropout=dropout,
                    )
                    for _ in range(num_blocks)
                ]
            )
        elif pos_encoding == "rope_2d":
            self.pos_embed = None
            head_dim = d_model // num_heads
            max_grid_side = self.max_img_size // patch_size
            self.rope_1d = None
            self.rope_2d = RoPE2D(head_dim, grid_size=max_grid_side)
            self.blocks = nn.ModuleList(
                [
                    RoPEBlock(
                        d_model,
                        num_heads,
                        pos_encoding="rope_2d",
                        rope_2d=self.rope_2d,
                        dropout=dropout,
                    )
                    for _ in range(num_blocks)
                ]
            )
        else:
            raise ValueError(f"unknown pos_encoding: {pos_encoding}")

        self.ln = nn.LayerNorm(d_model)

    def _positional_embedding(self, num_patches: int, grid_side: int) -> torch.Tensor:
        """Learned PE at eval resolution; bicubic-interpolate patch grid if needed."""
        assert self.pos_embed is not None
        if num_patches == self.train_num_patches:
            return self.pos_embed

        cls_pos = self.pos_embed[:, :1, :]
        patch_pos = self.pos_embed[:, 1:, :]
        patch_pos = patch_pos.reshape(
            1, self.train_grid_side, self.train_grid_side, self.d_model
        ).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(
            patch_pos,
            size=(grid_side, grid_side),
            mode="bicubic",
            align_corners=False,
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, num_patches, self.d_model)
        return torch.cat((cls_pos, patch_pos), dim=1)

    def _position_ids(
        self, num_patches: int, grid_side: int, device: torch.device
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if self.pos_encoding == "rope_1d":
            positions_1d = torch.arange(num_patches + 1, device=device)
            return positions_1d, None, None

        if self.pos_encoding == "rope_2d":
            patch_idx = torch.arange(num_patches, device=device)
            x_coords = torch.cat(
                (torch.zeros(1, device=device, dtype=torch.long), patch_idx % grid_side)
            )
            y_coords = torch.cat(
                (torch.zeros(1, device=device, dtype=torch.long), patch_idx // grid_side)
            )
            return None, x_coords, y_coords

        return None, None, None

    def forward(self, x: torch.Tensor, return_all_tokens=False) -> torch.Tensor:
        grid_side = x.shape[-1] // self.patch_size
        num_patches = grid_side * grid_side

        x = self.embeddings(x)

        # broadcast cls token to batch size for concat
        batch_size = x.shape[0]
        cls = self.cls_token.expand(batch_size, -1, -1)

        # append cls to front of sequence
        x = torch.cat((cls, x), dim=1)

        if self.pos_encoding == "learned":
            x = x + self._positional_embedding(num_patches, grid_side)
            for block in self.blocks:
                x = block(x)
        else:
            positions_1d, x_coords, y_coords = self._position_ids(
                num_patches, grid_side, x.device
            )
            for block in self.blocks:
                x = block(
                    x,
                    positions_1d=positions_1d,
                    x_coords=x_coords,
                    y_coords=y_coords,
                )

        x = self.ln(x)
        if return_all_tokens:
            return x
        return x[:, 0, :]

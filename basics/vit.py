"""Vision Transformer — §2.

You implement: PatchEmbeddings, ViT.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

import basics


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
    ) -> None:
        super().__init__()

        # shapes
        self.img_size = img_size
        self.patch_size = patch_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.dropout = dropout

        # params
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))
        self.embeddings = PatchEmbeddings(img_size, patch_size, d_model)
        self.blocks = nn.ModuleList(
            [
                basics.model.Block(
                    d_model, num_heads, num_patches + 1, is_decoder=False, dropout=dropout
                )
            ]
        )
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, return_all_tokens=False) -> torch.Tensor:
        x = self.embeddings(x)

        # broadcast cls token to batch size for concat
        batch_size = x.shape[0]
        cls = self.cls_token.expand(batch_size, -1, -1)

        # append cls to front of sequence
        x = torch.cat((cls, x), dim=1)

        x += self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        if return_all_tokens:
            return x
        return x[:, 0, :]

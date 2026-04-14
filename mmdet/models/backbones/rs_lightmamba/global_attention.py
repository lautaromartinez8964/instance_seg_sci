from __future__ import annotations

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath


class RSGlobalAttentionBlock(nn.Module):
    """Global self-attention block for the semantic final stage."""

    def __init__(self,
                 hidden_dim: int,
                 num_heads: int = 8,
                 mlp_ratio: float = 4.0,
                 drop_path: float = 0.0,
                 qkv_bias: bool = True,
                 attn_drop: float = 0.0,
                 proj_drop: float = 0.0,
                 act_layer: type[nn.Module] = nn.GELU,
                 use_checkpoint: bool = False,
                 channel_first: bool = False,
                 use_pos_embed: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.channel_first = channel_first
        self.use_checkpoint = use_checkpoint
        self.use_pos_embed = use_pos_embed
        self.pos_embed = None
        if self.use_pos_embed:
            self.pos_embed = nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=1,
                groups=hidden_dim,
                bias=True)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=attn_drop,
            bias=qkv_bias,
            batch_first=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.norm2 = nn.LayerNorm(hidden_dim)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(proj_drop),
            nn.Linear(mlp_hidden_dim, hidden_dim),
            nn.Dropout(proj_drop),
        )

    def _flatten(self, x: torch.Tensor):
        if self.channel_first:
            batch_size, channels, height, width = x.shape
            tokens = x.flatten(2).transpose(1, 2)
            return tokens, (batch_size, channels, height, width)

        batch_size, height, width, channels = x.shape
        tokens = x.reshape(batch_size, height * width, channels)
        return tokens, (batch_size, height, width, channels)

    def _restore(self, tokens: torch.Tensor, shape):
        if self.channel_first:
            batch_size, channels, height, width = shape
            return tokens.transpose(1, 2).reshape(batch_size, channels, height,
                                                  width)

        batch_size, height, width, channels = shape
        return tokens.reshape(batch_size, height, width, channels)

    def _apply_pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_pos_embed:
            return x

        if self.channel_first:
            return x + self.pos_embed(x)

        x_cf = x.permute(0, 3, 1, 2).contiguous()
        x_cf = x_cf + self.pos_embed(x_cf)
        return x_cf.permute(0, 2, 3, 1).contiguous()

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._apply_pos_embed(x)
        tokens, shape = self._flatten(x)

        attn_input = self.norm1(tokens)
        attn_out, _ = self.attn(attn_input, attn_input, attn_input,
                                need_weights=False)
        tokens = tokens + self.drop_path(attn_out)
        tokens = tokens + self.drop_path(self.mlp(self.norm2(tokens)))

        return self._restore(tokens, shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, x)
        return self._forward(x)
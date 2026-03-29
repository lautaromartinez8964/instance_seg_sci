from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ForegroundHead(nn.Module):
    """Lightweight foreground importance predictor for IG-Scan."""

    def __init__(self, d_inner: int):
        super().__init__()
        self.dw_conv = nn.Conv2d(
            d_inner,
            d_inner,
            kernel_size=3,
            padding=1,
            groups=d_inner,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(d_inner)
        self.act = nn.SiLU(inplace=True)
        self.pw_conv = nn.Conv2d(d_inner, 1, kernel_size=1, bias=True)
        nn.init.constant_(self.pw_conv.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw_conv(x)
        x = self.bn(x)
        x = self.act(x)
        return torch.sigmoid(self.pw_conv(x))


def compute_region_scores(importance: torch.Tensor, gh: int,
                          gw: int) -> torch.Tensor:
    scores = F.adaptive_avg_pool2d(importance, (gh, gw))
    return scores.flatten(1)


def _get_divisible_grid(height: int, width: int,
                        region_size: int) -> Tuple[int, int]:
    gh = max(height // region_size, 1)
    gw = max(width // region_size, 1)
    gh = min(gh, height)
    gw = min(gw, width)

    while gh > 1 and height % gh != 0:
        gh -= 1
    while gw > 1 and width % gw != 0:
        gw -= 1

    return gh, gw


def get_region_order(scores: torch.Tensor, descending: bool = True) -> torch.Tensor:
    try:
        return torch.argsort(
            scores, dim=-1, descending=descending, stable=True)
    except TypeError:
        eps = torch.arange(
            scores.size(-1), device=scores.device,
            dtype=scores.dtype).view(1, -1) * 1e-6
        adjusted = scores - eps if descending else scores + eps
        return torch.argsort(adjusted, dim=-1, descending=descending)


def region_permute_2d(x: torch.Tensor, order: torch.Tensor, gh: int,
                      gw: int) -> torch.Tensor:
    batch_size, channels, height, width = x.shape
    region_h = height // gh
    region_w = width // gw
    group_count = gh * gw
    region_area = region_h * region_w

    x = x.view(batch_size, channels, gh, region_h, gw, region_w)
    x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
    x = x.view(batch_size, channels, group_count, region_area)

    gather_index = order.unsqueeze(1).unsqueeze(-1).expand(
        batch_size, channels, group_count, region_area)
    x = torch.gather(x, dim=2, index=gather_index)

    x = x.view(batch_size, channels, gh, gw, region_h, region_w)
    x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
    return x.view(batch_size, channels, height, width)


def region_inv_permute_2d(y: torch.Tensor, order: torch.Tensor, gh: int,
                          gw: int) -> torch.Tensor:
    inverse_order = torch.argsort(order, dim=-1)
    return region_permute_2d(y, inverse_order, gh, gw)


class FGIGScan(nn.Module):
    """Foreground-guided region reordering in 2D space.

    Note:
        The hard sort path is non-differentiable. To keep the importance head
        trainable without changing MMDetection's detector interface, we add a
        lightweight differentiable foreground modulation before permutation.
    """

    def __init__(self,
                 d_inner: int,
                 region_size: int = 4,
                 guidance_scale: float = 0.5):
        super().__init__()
        self.fg_head = ForegroundHead(d_inner)
        self.region_size = region_size
        self.guidance_scale = nn.Parameter(torch.tensor(float(guidance_scale)))

        self._order: Optional[torch.Tensor] = None
        self._gh: int = 0
        self._gw: int = 0
        self.last_importance: Optional[torch.Tensor] = None

    def permute_2d(self,
                   x: torch.Tensor,
                   descending: bool = True) -> torch.Tensor:
        _, _, height, width = x.shape
        gh, gw = _get_divisible_grid(height, width, self.region_size)

        importance = self.fg_head(x)
        scores = compute_region_scores(importance, gh, gw)
        order = get_region_order(scores, descending=descending)

        self._order = order
        self._gh = gh
        self._gw = gw
        self.last_importance = importance

        x = x * (1 + self.guidance_scale * importance)
        return region_permute_2d(x, order, gh, gw)

    def inv_permute_2d(self, y: torch.Tensor) -> torch.Tensor:
        if self._order is None:
            raise RuntimeError(
                'FGIGScan.inv_permute_2d called before permute_2d.')
        return region_inv_permute_2d(y, self._order, self._gh, self._gw)
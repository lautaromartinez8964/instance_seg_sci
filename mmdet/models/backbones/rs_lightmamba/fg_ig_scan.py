from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _resolve_group_count(num_channels: int, max_groups: int = 8) -> int:
    max_groups = max(1, min(max_groups, num_channels))
    for group_count in range(max_groups, 0, -1):
        if num_channels % group_count == 0:
            return group_count
    return 1


def _stable_argsort(scores: torch.Tensor, descending: bool) -> torch.Tensor:
    try:
        return torch.argsort(
            scores, dim=-1, descending=descending, stable=True)
    except TypeError:
        group_count = scores.size(-1)
        eps = torch.arange(
            group_count, device=scores.device,
            dtype=scores.dtype).view(1, -1) * 1e-6
        adjusted = scores - eps if descending else scores + eps
        return torch.argsort(adjusted, dim=-1, descending=descending)


class ForegroundHead(nn.Module):
    """Large-kernel bottleneck importance predictor."""

    def __init__(self,
                 d_inner: int,
                 lk_size: int = 7,
                 reduction: int = 4,
                 norm_type: str = 'bn',
                 gn_groups: int = 8):
        super().__init__()
        mid_channels = max(d_inner // reduction, 16)
        norm_type = norm_type.lower()
        if norm_type not in {'bn', 'gn'}:
            raise ValueError(f'Unsupported norm_type: {norm_type}')

        def build_norm() -> nn.Module:
            if norm_type == 'gn':
                return nn.GroupNorm(_resolve_group_count(mid_channels, gn_groups),
                                    mid_channels)
            return nn.BatchNorm2d(mid_channels)

        self.pw_down = nn.Conv2d(d_inner, mid_channels, 1, bias=False)
        self.bn1 = build_norm()
        self.dw_large = nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=lk_size,
            padding=lk_size // 2,
            groups=mid_channels,
            bias=False,
        )
        self.bn2 = build_norm()
        self.pw_mix = nn.Conv2d(mid_channels, mid_channels, 1, bias=False)
        self.bn3 = build_norm()
        self.dw_refine = nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=3,
            padding=1,
            groups=mid_channels,
            bias=False,
        )
        self.bn4 = build_norm()
        self.pw_out = nn.Conv2d(mid_channels, 1, 1, bias=True)
        nn.init.constant_(self.pw_out.bias, 0.0)
        self.act = nn.SiLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.pw_down(x)))
        x = self.act(self.bn2(self.dw_large(x)))
        x = self.act(self.bn3(self.pw_mix(x)))
        x = self.act(self.bn4(self.dw_refine(x)))
        return torch.sigmoid(self.pw_out(x))


def compute_region_scores(importance: torch.Tensor, gh: int,
                          gw: int) -> torch.Tensor:
    scores = F.adaptive_avg_pool2d(importance, (gh, gw))
    return scores.flatten(1)


def _get_divisible_grid(height: int, width: int,
                        region_size: int) -> Tuple[int, int]:
    if region_size <= 0:
        raise ValueError('region_size must be a positive integer.')
    if height <= 0 or width <= 0:
        raise ValueError('height and width must be positive integers.')

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
    return _stable_argsort(scores, descending=descending)


class FGIGScan(nn.Module):
    """Foreground-guided region ordering for 1D scan construction."""

    def __init__(self,
                 d_inner: int,
                 region_size: int = 4,
                 guidance_scale: float = 0.1,
                 lk_size: int = 7,
                 fg_loss_weight: float = 0.0,
                 fg_norm_type: str = 'bn',
                 fg_gn_groups: int = 8):
        super().__init__()
        self.fg_head = ForegroundHead(
            d_inner,
            lk_size=lk_size,
            norm_type=fg_norm_type,
            gn_groups=fg_gn_groups)
        self.region_size = region_size
        self.fg_loss_weight = float(max(fg_loss_weight, 0.0))
        guidance_scale = float(max(min(guidance_scale, 1 - 1e-4), 1e-4))
        self._guidance_scale_raw = nn.Parameter(
            torch.logit(torch.tensor(guidance_scale)))

        self.current_fg_target: Optional[torch.Tensor] = None
        self.last_importance: Optional[torch.Tensor] = None
        self.last_order: Optional[torch.Tensor] = None
        self.last_region_scores: Optional[torch.Tensor] = None
        self.last_grid_shape: Optional[Tuple[int, int]] = None
        self.last_fg_loss: Optional[torch.Tensor] = None

    @property
    def guidance_scale(self) -> torch.Tensor:
        return torch.sigmoid(self._guidance_scale_raw)

    def reset_state(self) -> None:
        self.last_importance = None
        self.last_order = None
        self.last_region_scores = None
        self.last_grid_shape = None
        self.last_fg_loss = None

    def set_fg_target(self, fg_target: Optional[torch.Tensor]) -> None:
        self.current_fg_target = fg_target

    def clear_fg_target(self) -> None:
        self.current_fg_target = None

    def predict_importance(self, x: torch.Tensor) -> torch.Tensor:
        importance = self.fg_head(x)
        self.last_fg_loss = self._compute_fg_loss(importance)
        self.last_importance = importance
        self.last_order = None
        self.last_region_scores = None
        self.last_grid_shape = None
        return importance

    def _compute_fg_loss(self, importance: torch.Tensor) -> Optional[torch.Tensor]:
        if self.current_fg_target is None or self.fg_loss_weight <= 0:
            return None

        fg_target = self.current_fg_target.to(
            device=importance.device, dtype=importance.dtype)
        if fg_target.shape[-2:] != importance.shape[-2:]:
            fg_target = F.interpolate(
                fg_target,
                size=importance.shape[-2:],
                mode='area')
            fg_target = (fg_target > 0).to(dtype=importance.dtype)

        # BCE on probabilities is not autocast-safe. Compute it in FP32 and
        # return the scaled scalar to the mixed-precision training loop.
        with torch.amp.autocast(device_type=importance.device.type, enabled=False):
            loss = F.binary_cross_entropy(
                importance.float(),
                fg_target.float())
        return loss * self.fg_loss_weight

    def compute_region_order(
        self,
        x: torch.Tensor,
        descending: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f'x must be a 4D tensor, got shape {tuple(x.shape)}')

        _, _, height, width = x.shape
        gh, gw = _get_divisible_grid(height, width, self.region_size)
        importance = self.predict_importance(x)
        scores = compute_region_scores(importance, gh, gw)
        order = get_region_order(scores, descending=descending)

        self.last_importance = importance
        self.last_order = order
        self.last_region_scores = scores
        self.last_grid_shape = (gh, gw)

        x_mod = x * (1.0 + self.guidance_scale * importance)
        return x_mod, order, gh, gw, importance
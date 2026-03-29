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
        self.act = nn.SiLU(inplace=False)
        self.pw_conv = nn.Conv2d(d_inner, 1, kernel_size=1, bias=True)
        nn.init.constant_(self.pw_conv.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw_conv(x)
        x = self.bn(x)
        x = self.act(x)
        return torch.sigmoid(self.pw_conv(x))


def compute_region_scores(importance: torch.Tensor, gh: int,
                          gw: int) -> torch.Tensor:
    """
    通过自适应平均池化计算区域重要性分数
    
    将输入的重要性图池化到指定的网格大小，然后展平为二维张量。
    
    Args:
        importance (torch.Tensor): 输入的重要性图，形状为 (B, C, H, W)
        gh (int): 目标网格高度
        gw (int): 目标网格宽度
    
    Returns:
        torch.Tensor: 展平后的区域分数，形状为 (B, C * gh * gw)
    """
    scores = F.adaptive_avg_pool2d(importance, (gh, gw))
    return scores.flatten(1)


def _get_divisible_grid(height: int, width: int,
                        region_size: int) -> Tuple[int, int]:
    """计算特征图的可整除网格划分方案。
    
    该函数根据给定的区域大小，将特征图划分为多个网格，并确保网格数量
    能够整除特征图的尺寸，以便后续的区域重排序操作能够正确进行。
    
    Args:
        height (int): 特征图的高度
        width (int): 特征图的宽度
        region_size (int): 期望的区域大小（每个区域的高度/宽度）
    
    Returns:
        Tuple[int, int]: 可整除的网格数量 (gh, gw)，其中 gh 是高度方向的网格数，
                         gw 是宽度方向的网格数
    """
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
    """
    根据区域重要性分数获取排序索引
    
    对于相同分数的区域，通过添加微小扰动确保排序的确定性。
    当 PyTorch 版本不支持 stable 参数时，回退到扰动方案。
    
    Args:
        scores (torch.Tensor): 区域重要性分数，形状为 (batch_size, num_regions)
        descending (bool, optional): 是否按降序排序。默认为 True
    
    Returns:
        torch.Tensor: 排序索引，形状与 scores 相同
    
    Raises:
        TypeError: 当 scores 的数据类型不支持 arange 操作时（理论上不应触发）
    """
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
    """
    使用区域重要性顺序对输入张量进行 2D 空间重排列。
    
    该函数根据提供的区域顺序（order），将特征图划分为 gh × gw 个区域，
    并按照重要性顺序重新排列这些区域。支持前向排列和反向排列。
    
    Args:
        x (torch.Tensor): 输入特征图，形状为 (batch_size, channels, height, width)
        order (torch.Tensor): 区域重排顺序，形状为 (batch_size, gh * gw)
        gh (int): 高度方向的网格数量
        gw (int): 宽度方向的网格数量
    
    Returns:
        torch.Tensor: 重排后的特征图，形状与输入相同 (batch_size, channels, height, width)
    
    Raises:
        RuntimeError: 如果 height 不能被 gh 整除，或 width 不能被 gw 整除
        ValueError: 如果 order 的形状与 (batch_size, gh * gw) 不匹配
    
    Note:
        该操作是非可微的，仅用于推理阶段或作为硬排序路径。
        在训练阶段建议使用可微的调制机制。
    """
    if x.ndim != 4:
        raise ValueError(f'x must be a 4D tensor, got shape {tuple(x.shape)}')
    if order.ndim != 2:
        raise ValueError(
            f'order must be a 2D tensor, got shape {tuple(order.shape)}')

    batch_size, channels, height, width = x.shape
    if order.shape[0] != batch_size or order.shape[1] != gh * gw:
        raise ValueError(
            'order shape must be (batch_size, gh * gw), got '
            f'{tuple(order.shape)} vs ({batch_size}, {gh * gw})')
    if height % gh != 0 or width % gw != 0:
        raise ValueError(
            f'Feature map {(height, width)} is not divisible by grid {(gh, gw)}')

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
    """对 2D 特征图执行区域逆排列操作。

    根据之前计算的排列顺序，将经过区域重排的特征图恢复到原始空间布局。
    这是 `region_permute_2d` 的逆操作，用于确保前向传播和反向传播的一致性。

    Args:
        y (torch.Tensor): 输入的特征图张量，形状为 (batch_size, channels, height, width)。
        order (torch.Tensor): 区域排列顺序索引，形状为 (batch_size, gh * gw)，
                             由 `region_permute_2d` 或 `get_region_order` 生成。
        gh (int): 高度方向的网格划分数量，必须满足 height % gh == 0。
        gw (int): 宽度方向的网格划分数量，必须满足 width % gw == 0。

    Returns:
        torch.Tensor: 逆排列后的特征图张量，形状与输入 y 相同。

    Raises:
        RuntimeError: 如果 `order` 的形状与 (batch_size, gh * gw) 不匹配。
        ValueError: 如果输入张量的维度不是 4D。
    """
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
        guidance_scale = float(max(min(guidance_scale, 1 - 1e-4), 1e-4))
        self._guidance_scale_raw = nn.Parameter(
            torch.logit(torch.tensor(guidance_scale)))

        self._order: Optional[torch.Tensor] = None
        self._gh: int = 0
        self._gw: int = 0
        self.last_importance: Optional[torch.Tensor] = None

    @property
    def guidance_scale(self) -> torch.Tensor:
        """Constrained guidance scale in [0, 1] range via sigmoid."""
        return torch.sigmoid(self._guidance_scale_raw)

    def reset_state(self) -> None:
        """Reset cached state for clean inference on new inputs."""
        self._order = None
        self._gh = 0
        self._gw = 0
        self.last_importance = None

    def permute_2d(self,
                   x: torch.Tensor,
                   descending: bool = True) -> torch.Tensor:
        """基于前景重要性对特征图进行区域重排序。
        
        该方法首先计算特征图的前景重要性分数，然后将特征图划分为可整除的区域网格，
        根据每个区域的重要性分数对区域进行排序，最终按重要性顺序重排特征图中的区域。
        同时应用前景引导的调制增强重要区域的特征响应。
        
        Args:
            x (torch.Tensor): 输入特征图，形状为 (batch_size, channels, height, width)
            descending (bool, optional): 是否按重要性降序排列区域。默认为 True，
                即重要性高的区域排在前面。
        
        Returns:
            torch.Tensor: 重排序后的特征图，形状与输入相同 (batch_size, channels, height, width)。
                特征图中的区域已按照重要性分数重新排列，并应用了前景调制。
        
        Raises:
            RuntimeError: 如果在调用此方法前未正确初始化相关内部状态（如 _order、_gh 等）。
        """
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
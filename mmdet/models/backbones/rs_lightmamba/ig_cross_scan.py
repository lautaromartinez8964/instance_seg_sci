from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


def _reshape_regions(x: torch.Tensor, gh: int,
                     gw: int) -> Tuple[torch.Tensor, int, int]:
    batch_size, channels, height, width = x.shape
    if height % gh != 0 or width % gw != 0:
        raise ValueError(
            f'Feature map {(height, width)} is not divisible by grid {(gh, gw)}')
    region_h = height // gh
    region_w = width // gw
    x = x.view(batch_size, channels, gh, region_h, gw, region_w)
    x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
    return x, region_h, region_w


def _row_major_regions(x_regions: torch.Tensor) -> torch.Tensor:
    batch_size, channels, gh, gw, region_h, region_w = x_regions.shape
    return x_regions.view(batch_size, channels, gh * gw, region_h * region_w)


def _col_major_regions(x_regions: torch.Tensor) -> torch.Tensor:
    batch_size, channels, gh, gw, region_h, region_w = x_regions.shape
    x_regions = x_regions.permute(0, 1, 2, 4, 3, 5).contiguous()
    x_regions = x_regions.view(batch_size, channels, gh, region_h, gw, region_w)
    x_regions = x_regions.permute(0, 1, 2, 4, 5, 3).contiguous()
    return x_regions.view(batch_size, channels, gh * gw, region_h * region_w)


def _gather_regions(tokens: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    batch_size, channels, group_count, region_area = tokens.shape
    gather_index = order.unsqueeze(1).unsqueeze(-1).expand(
        batch_size, channels, group_count, region_area)
    return torch.gather(tokens, dim=2, index=gather_index)


def _restore_row_major(tokens: torch.Tensor, order: torch.Tensor, gh: int,
                       gw: int, region_h: int, region_w: int) -> torch.Tensor:
    batch_size, channels, group_count, region_area = tokens.shape
    restored = torch.zeros_like(tokens)
    scatter_index = order.unsqueeze(1).unsqueeze(-1).expand(
        batch_size, channels, group_count, region_area)
    restored.scatter_(2, scatter_index, tokens)
    restored = restored.view(batch_size, channels, gh, gw, region_h, region_w)
    restored = restored.permute(0, 1, 2, 4, 3, 5).contiguous()
    return restored.view(batch_size, channels, gh * region_h, gw * region_w)


def _restore_col_major(tokens: torch.Tensor, order: torch.Tensor, gh: int,
                       gw: int, region_h: int, region_w: int) -> torch.Tensor:
    batch_size, channels, group_count, region_area = tokens.shape
    restored = torch.zeros_like(tokens)
    scatter_index = order.unsqueeze(1).unsqueeze(-1).expand(
        batch_size, channels, group_count, region_area)
    restored.scatter_(2, scatter_index, tokens)
    restored = restored.view(batch_size, channels, gh, gw, region_w, region_h)
    restored = restored.permute(0, 1, 2, 3, 5, 4).contiguous()
    restored = restored.permute(0, 1, 2, 4, 3, 5).contiguous()
    return restored.view(batch_size, channels, gh * region_h, gw * region_w)


def ig_cross_scan_fn(
    x: torch.Tensor,
    order: torch.Tensor,
    gh: int,
    gw: int,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor | int]]:
    batch_size, channels, height, width = x.shape
    group_count = gh * gw
    if order.shape != (batch_size, group_count):
        raise ValueError(
            f'order shape must be {(batch_size, group_count)}, got {tuple(order.shape)}')

    x_regions, region_h, region_w = _reshape_regions(x, gh, gw)
    row_tokens = _row_major_regions(x_regions)
    col_tokens = _col_major_regions(x_regions)

    row_sorted = _gather_regions(row_tokens, order).reshape(batch_size, channels,
                                                            height * width)
    col_sorted = _gather_regions(col_tokens, order).reshape(batch_size, channels,
                                                            height * width)

    xs = torch.stack([
        row_sorted,
        col_sorted,
        row_sorted.flip(dims=[-1]),
        col_sorted.flip(dims=[-1]),
    ], dim=1)

    scan_info: Dict[str, torch.Tensor | int] = {
        'order': order,
        'gh': gh,
        'gw': gw,
        'region_h': region_h,
        'region_w': region_w,
    }
    return xs, scan_info


def restore_ig_path(path: torch.Tensor,
                    scan_info: Dict[str, torch.Tensor | int],
                    path_type: str,
                    reverse: bool = False) -> torch.Tensor:
    order = scan_info['order']
    gh = int(scan_info['gh'])
    gw = int(scan_info['gw'])
    region_h = int(scan_info['region_h'])
    region_w = int(scan_info['region_w'])

    if reverse:
        path = path.flip(dims=[-1])
    path = path.reshape(path.shape[0], path.shape[1], gh * gw, region_h * region_w)
    if path_type == 'row':
        return _restore_row_major(path, order, gh, gw, region_h, region_w)
    if path_type == 'col':
        return _restore_col_major(path, order, gh, gw, region_h, region_w)
    raise ValueError(f'Unsupported path_type: {path_type}')


def ig_cross_merge_fn(
    ys: torch.Tensor,
    scan_info: Dict[str, torch.Tensor | int],
) -> torch.Tensor:
    y0 = restore_ig_path(ys[:, 0], scan_info, path_type='row', reverse=False)
    y1 = restore_ig_path(ys[:, 1], scan_info, path_type='col', reverse=False)
    y2 = restore_ig_path(ys[:, 2], scan_info, path_type='row', reverse=True)
    y3 = restore_ig_path(ys[:, 3], scan_info, path_type='col', reverse=True)
    return y0 + y1 + y2 + y3


class IGCrossScan(nn.Module):
    def __init__(self):
        super().__init__()
        self._scan_info: Dict[str, torch.Tensor | int] = {}

    def reset_state(self) -> None:
        self._scan_info = {}

    def scan(self, x: torch.Tensor, order: torch.Tensor, gh: int,
             gw: int) -> torch.Tensor:
        xs, self._scan_info = ig_cross_scan_fn(x, order, gh, gw)
        return xs

    def merge(self, ys: torch.Tensor) -> torch.Tensor:
        if not self._scan_info:
            raise RuntimeError('IGCrossScan.merge called before scan.')
        y = ig_cross_merge_fn(ys, self._scan_info)
        self.reset_state()
        return y
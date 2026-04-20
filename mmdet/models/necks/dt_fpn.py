# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmdet.registry import MODELS

from .fpn import FPN


@MODELS.register_module()
class DTFPN(FPN):
    """FPN with distance-transform-guided top-down fusion.

    The deepest lateral feature predicts a normalized distance transform map.
    During top-down fusion, each level uses the DT map to interpolate between
    its local lateral feature and the upsampled deeper feature.
    """

    def __init__(self,
                 *args,
                 guided_levels: List[int] | None = None,
                 dt_head_channels: int = 128,
                 dt_loss_weight: float = 0.2,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        num_fusions = self.backbone_end_level - self.start_level - 1
        if guided_levels is None:
            guided_levels = list(range(num_fusions))
        self.guided_levels = sorted(set(guided_levels))
        invalid_levels = [
            level for level in self.guided_levels
            if level < 0 or level >= num_fusions
        ]
        if invalid_levels:
            raise ValueError(
                f'guided_levels contains invalid indices: {invalid_levels}.')

        self.dt_head = nn.Sequential(
            nn.Conv2d(self.out_channels, dt_head_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dt_head_channels, 1, kernel_size=1))
        self.gate_scales = nn.Parameter(torch.ones(num_fusions))
        self.gate_biases = nn.Parameter(torch.zeros(num_fusions))
        self._dt_loss_weight = float(max(dt_loss_weight, 0.0))

        self._current_dt_target: torch.Tensor | None = None
        self._last_dt_map: Tensor | None = None
        self._last_dt_loss: Tensor | None = None
        self.last_gates: Tuple[Tensor, ...] = ()

    def _extract_dt_target(self, data_sample, device: torch.device) -> torch.Tensor | None:
        if 'gt_sem_seg' not in data_sample:
            return None
        dt_target = data_sample.gt_sem_seg.sem_seg.to(device=device, dtype=torch.float32)
        if dt_target.ndim == 2:
            dt_target = dt_target.unsqueeze(0)
        elif dt_target.ndim == 4 and dt_target.shape[0] == 1:
            dt_target = dt_target.squeeze(0)
        if dt_target.ndim == 3 and dt_target.shape[-1] == 1 and dt_target.shape[0] != 1:
            dt_target = dt_target.permute(2, 0, 1).contiguous()
        dt_target = dt_target[:1]
        if dt_target.max() > 1.0:
            dt_target = dt_target / 255.0
        return dt_target.clamp_(0.0, 1.0)

    def set_auxiliary_targets(self, batch_data_samples, device=None) -> None:
        device = device or next(self.parameters()).device
        targets = []
        for data_sample in batch_data_samples:
            dt_target = self._extract_dt_target(data_sample, device)
            if dt_target is None:
                self.clear_auxiliary_targets()
                return
            targets.append(dt_target)
        self._current_dt_target = torch.stack(targets, dim=0) if targets else None
        self._last_dt_map = None
        self._last_dt_loss = None

    def clear_auxiliary_targets(self) -> None:
        self._current_dt_target = None
        self._last_dt_map = None
        self._last_dt_loss = None
        self.last_gates = ()

    def _compute_dt_loss(self, dt_map: Tensor) -> Tensor | None:
        if self._current_dt_target is None or self._dt_loss_weight <= 0:
            return None
        dt_target = self._current_dt_target.to(device=dt_map.device, dtype=dt_map.dtype)
        if dt_target.shape[-2:] != dt_map.shape[-2:]:
            dt_target = F.interpolate(
                dt_target,
                size=dt_map.shape[-2:],
                mode='bilinear',
                align_corners=False)
        with torch.amp.autocast(device_type=dt_map.device.type, enabled=False):
            loss = F.mse_loss(dt_map.float(), dt_target.float())
        return loss * self._dt_loss_weight

    def get_auxiliary_losses(self) -> dict:
        loss = self._last_dt_loss
        self.clear_auxiliary_targets()
        if loss is None:
            return {}
        return {'loss_neck_dt': loss}

    def get_last_dt_map(self) -> Tensor | None:
        return self._last_dt_map

    def forward(self, inputs) -> tuple:
        backbone_inputs = tuple(inputs)
        assert len(backbone_inputs) == len(self.in_channels)

        laterals = [
            lateral_conv(backbone_inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        dt_map = torch.sigmoid(self.dt_head(laterals[-1]))
        self._last_dt_map = dt_map
        self._last_dt_loss = self._compute_dt_loss(dt_map)

        used_backbone_levels = len(laterals)
        gate_records: List[Tensor] = []
        for i in range(used_backbone_levels - 1, 0, -1):
            target_level = i - 1
            if 'scale_factor' in self.upsample_cfg:
                upsampled = F.interpolate(laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[target_level].shape[2:]
                upsampled = F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

            if target_level in self.guided_levels:
                resized_dt = dt_map
                if resized_dt.shape[2:] != laterals[target_level].shape[2:]:
                    resized_dt = F.interpolate(
                        resized_dt,
                        size=laterals[target_level].shape[2:],
                        mode='bilinear',
                        align_corners=False)
                scale = self.gate_scales[target_level].to(dtype=upsampled.dtype).view(1, 1, 1, 1)
                bias = self.gate_biases[target_level].to(dtype=upsampled.dtype).view(1, 1, 1, 1)
                gate = torch.sigmoid(scale * resized_dt.to(dtype=upsampled.dtype) + bias)
                laterals[target_level] = (
                    laterals[target_level] * (1.0 - gate) + upsampled * gate)
                gate_records.insert(0, gate.detach())
            else:
                laterals[target_level] = laterals[target_level] + upsampled

        self.last_gates = tuple(gate_records)

        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        if self.num_outs > len(outs):
            if not self.add_extra_convs:
                for _ in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = backbone_inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
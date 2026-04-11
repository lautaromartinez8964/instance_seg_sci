# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from mmdet.registry import MODELS

from .fpn import FPN


@MODELS.register_module()
class HF_FPN(FPN):
    """FPN with high-frequency guided top-down fusion."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        num_fusions = self.backbone_end_level - self.start_level - 1
        self.fusion_beta = torch.nn.Parameter(torch.zeros(num_fusions))
        self.last_hf_maps: Tuple[Tensor, ...] = ()
        self.last_gates: Tuple[Tensor, ...] = ()

    def _split_inputs(self, inputs):
        if (isinstance(inputs, tuple) and len(inputs) == 2
                and isinstance(inputs[0], (list, tuple))
                and isinstance(inputs[1], (list, tuple))):
            return tuple(inputs[0]), tuple(inputs[1])
        return tuple(inputs), None

    def forward(self, inputs) -> tuple:
        """Forward with optional `(features, hf_maps)` backbone output."""
        backbone_inputs, hf_maps = self._split_inputs(inputs)
        assert len(backbone_inputs) == len(self.in_channels)

        expected_hf_maps = self.backbone_end_level - self.start_level - 1
        if hf_maps is not None and len(hf_maps) != expected_hf_maps:
            raise ValueError(
                f'HF_FPN expects {expected_hf_maps} hf maps, got {len(hf_maps)}.')

        self.last_hf_maps = tuple(hf_maps) if hf_maps is not None else ()

        laterals = [
            lateral_conv(backbone_inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        used_backbone_levels = len(laterals)
        gate_records: List[Tensor] = []
        for i in range(used_backbone_levels - 1, 0, -1):
            if 'scale_factor' in self.upsample_cfg:
                upsampled = F.interpolate(laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                upsampled = F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

            if hf_maps is not None:
                hf_map = hf_maps[i - 1]
                if hf_map.shape[2:] != laterals[i - 1].shape[2:]:
                    hf_map = F.interpolate(
                        hf_map,
                        size=laterals[i - 1].shape[2:],
                        mode='bilinear',
                        align_corners=False)
                beta = self.fusion_beta[i - 1].to(dtype=upsampled.dtype).view(
                    1, 1, 1, 1)
                gate = 1.0 + beta * hf_map.to(dtype=upsampled.dtype)
                laterals[i - 1] = laterals[i - 1] + upsampled * gate
                gate_records.insert(0, gate.detach())
            else:
                laterals[i - 1] = laterals[i - 1] + upsampled

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
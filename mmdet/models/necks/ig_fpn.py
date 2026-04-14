# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from mmdet.registry import MODELS

from .fpn import FPN


@MODELS.register_module()
class IG_FPN(FPN):
    """FPN with a shared importance map gating top-down fusion."""

    def __init__(self, guided_levels: List[int] | None = None, *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        num_fusions = self.backbone_end_level - self.start_level - 1
        self.guided_levels = list(guided_levels) if guided_levels is not None else list(range(num_fusions))
        self.ig_alphas = torch.nn.Parameter(torch.zeros(num_fusions))
        # Keep legacy name for checkpoint compatibility and external probes.
        self.alpha = self.ig_alphas
        self.last_guidance_map: Tensor | None = None
        self.last_gates: Tuple[Tensor, ...] = ()

    def _split_inputs(self, inputs):
        if (isinstance(inputs, tuple) and len(inputs) == 2
                and isinstance(inputs[0], (list, tuple))
                and isinstance(inputs[1], torch.Tensor)):
            return tuple(inputs[0]), inputs[1]
        return tuple(inputs), None

    def forward(self, inputs) -> tuple:
        backbone_inputs, guidance_map = self._split_inputs(inputs)
        assert len(backbone_inputs) == len(self.in_channels)

        self.last_guidance_map = guidance_map.detach() if guidance_map is not None else None

        laterals = [
            lateral_conv(backbone_inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

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

            gate = None
            if guidance_map is not None and target_level in self.guided_levels:
                resized_guidance = guidance_map
                if resized_guidance.shape[2:] != laterals[target_level].shape[2:]:
                    resized_guidance = F.interpolate(
                        resized_guidance,
                        size=laterals[target_level].shape[2:],
                        mode='bilinear',
                        align_corners=False)
                alpha = F.softplus(self.ig_alphas[target_level]).to(dtype=upsampled.dtype).view(1, 1, 1, 1)
                gate = 1.0 + alpha * resized_guidance.to(dtype=upsampled.dtype)
                laterals[target_level] = laterals[target_level] + upsampled * gate
            else:
                laterals[target_level] = laterals[target_level] + upsampled

            if gate is not None:
                gate_records.insert(0, gate.detach())

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
# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmdet.registry import MODELS
from .fpn import FPN


@MODELS.register_module()
class IGFPN(FPN):
    """Importance-Guided Feature Pyramid Network (IG-FPN).

    Extends :class:`FPN` by accepting an extra ``importance_map`` produced by
    the backbone (typically fused from Stage 3 and Stage 4 features) and using
    it to modulate each top-down fusion step via:

    .. math::
        P_{i-1} = L_{i-1} + \\text{Up}(P_i) \\cdot (1 + \\alpha_i \\cdot
        \\text{Imp}_{i-1})

    where :math:`\\alpha_i` is a per-level learnable scalar initialised to
    ``0.0``, so the module degrades to a standard FPN at the start of
    training.

    Args:
        alpha_init (float): Initial value for every learnable alpha scalar.
            Defaults to ``0.0``.
        **kwargs: All remaining keyword arguments are forwarded to
            :class:`FPN`.

    Note:
        The backbone must append the importance map as the **last** element of
        its output tuple, *after* the regular feature maps.  The importance
        map should have shape ``(B, 1, H, W)`` with values in ``[0, 1]``
        (i.e. passed through a sigmoid).  The first ``len(in_channels)``
        elements of the backbone output are treated as the standard FPN
        inputs.
    """

    def __init__(self, alpha_init: float = 0.0, **kwargs) -> None:
        super().__init__(**kwargs)
        # Number of top-down fusion steps equals (num_levels - 1).
        # Each step updates laterals[i-1] using the gated upsampled laterals[i],
        # so we need one alpha per target lateral level (i-1 goes 0 .. N-2).
        num_levels = self.backbone_end_level - self.start_level
        self.alphas = nn.ParameterList([
            nn.Parameter(torch.tensor(float(alpha_init)))
            for _ in range(max(num_levels - 1, 0))
        ])

    def forward(self, inputs: Tuple[Tensor]) -> tuple:
        """Forward function.

        Args:
            inputs (tuple[Tensor]): The backbone output.  The **last**
                element must be the importance map ``(B, 1, H_imp, W_imp)``;
                the preceding ``len(self.in_channels)`` elements are the
                standard feature maps fed into the FPN lateral convolutions.

        Returns:
            tuple[Tensor]: FPN output feature maps, each a 4-D tensor.
        """
        assert len(inputs) == len(self.in_channels) + 1, (
            f'IGFPN expects {len(self.in_channels) + 1} inputs '
            f'(backbone features + importance_map), got {len(inputs)}')

        # Split importance map from backbone feature maps.
        imp_map = inputs[-1]          # (B, 1, H_imp, W_imp)
        feat_inputs = inputs[:-1]     # (C2, C3, C4, C5)

        # ------------------------------------------------------------------
        # Build laterals (identical to FPN)
        # ------------------------------------------------------------------
        laterals = [
            lateral_conv(feat_inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # ------------------------------------------------------------------
        # Build top-down path with importance gating
        # ------------------------------------------------------------------
        used_backbone_levels = len(laterals)

        # Pre-compute importance map at each target resolution to avoid
        # redundant interpolations inside the loop.
        imp_per_level = {}
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            if prev_shape not in imp_per_level:
                imp_per_level[prev_shape] = F.interpolate(
                    imp_map, size=prev_shape,
                    mode='bilinear', align_corners=False)

        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]

            # Upsample higher-level feature (same logic as parent FPN).
            if 'scale_factor' in self.upsample_cfg:
                upsampled = F.interpolate(laterals[i], **self.upsample_cfg)
            else:
                upsampled = F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

            # Gate: upsampled features are amplified in foreground regions.
            # alpha_idx = i - 1  (ranges from 0 at the lowest level up to N-2).
            # Clamp alpha to [-5, 5] to prevent numerical instability while
            # still allowing the module to learn strong modulation.
            alpha = torch.clamp(self.alphas[i - 1], min=-5.0, max=5.0)
            gate = 1.0 + alpha * imp_per_level[prev_shape]
            laterals[i - 1] = laterals[i - 1] + upsampled * gate

        # ------------------------------------------------------------------
        # Build outputs – part 1: from original backbone levels (same as FPN)
        # ------------------------------------------------------------------
        outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels)
        ]

        # ------------------------------------------------------------------
        # Build outputs – part 2: extra levels (identical to FPN)
        # ------------------------------------------------------------------
        if self.num_outs > len(outs):
            if not self.add_extra_convs:
                for _ in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = feat_inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(
                    self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))

        return tuple(outs)

# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, MultiConfig, OptConfigType
from .fpn import FPN


@MODELS.register_module()
class IGFPN(FPN):
    """Importance-Guided Feature Pyramid Network (IG-FPN).

    Extends the standard FPN by incorporating an importance map (produced by
    the backbone, e.g. from fused Stage-3 and Stage-4 features) into the
    top-down fusion path.  At each merging step the upsampled higher-level
    feature is scaled by ``(1 + alpha_i * importance_map_resized)`` before
    being added to the lateral feature, so that foreground-rich regions
    receive stronger guidance from the semantic top level.

    The importance map is expected to be the **last** element of the tuple
    returned by the backbone, i.e.::

        backbone_outputs = (C2, C3, C4, C5, imp_map)

    where ``imp_map`` has shape ``(B, 1, H3, W3)`` and values in ``[0, 1]``
    (sigmoid output).

    Args:
        in_channels (list[int]): Number of input channels per scale
            (excluding the importance map channel).
        out_channels (int): Number of output channels at each scale.
        num_outs (int): Number of output scales.
        alpha_init (float): Initial value for the per-level learnable
            scalar gate weight.  Defaults to ``0.0`` so that the module
            degrades to a standard FPN at initialisation.
        start_level (int): See :class:`FPN`.
        end_level (int): See :class:`FPN`.
        add_extra_convs (bool | str): See :class:`FPN`.
        relu_before_extra_convs (bool): See :class:`FPN`.
        no_norm_on_lateral (bool): See :class:`FPN`.
        conv_cfg (:obj:`ConfigDict` or dict, optional): See :class:`FPN`.
        norm_cfg (:obj:`ConfigDict` or dict, optional): See :class:`FPN`.
        act_cfg (:obj:`ConfigDict` or dict, optional): See :class:`FPN`.
        upsample_cfg (:obj:`ConfigDict` or dict): See :class:`FPN`.
        init_cfg (:obj:`ConfigDict` or dict or list): See :class:`FPN`.
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        num_outs: int,
        alpha_init: float = 0.0,
        start_level: int = 0,
        end_level: int = -1,
        add_extra_convs: Union[bool, str] = False,
        relu_before_extra_convs: bool = False,
        no_norm_on_lateral: bool = False,
        conv_cfg: OptConfigType = None,
        norm_cfg: OptConfigType = None,
        act_cfg: OptConfigType = None,
        upsample_cfg: ConfigType = dict(mode='nearest'),
        init_cfg: MultiConfig = dict(
            type='Xavier', layer='Conv2d', distribution='uniform')
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_outs=num_outs,
            start_level=start_level,
            end_level=end_level,
            add_extra_convs=add_extra_convs,
            relu_before_extra_convs=relu_before_extra_convs,
            no_norm_on_lateral=no_norm_on_lateral,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            upsample_cfg=upsample_cfg,
            init_cfg=init_cfg)

        # One learnable scalar alpha per top-down merging step.
        # There are (used_backbone_levels - 1) merging steps; we create a
        # parameter for each lateral level that *receives* a top-down signal,
        # i.e. levels [start_level .. backbone_end_level - 2] (indices 0 ..
        # used_backbone_levels - 2 in the laterals list).
        used_backbone_levels = self.backbone_end_level - self.start_level
        self.alphas = nn.ParameterList([
            nn.Parameter(torch.full((), alpha_init))
            for _ in range(used_backbone_levels - 1)
        ])

    def forward(self, inputs: Tuple[Tensor]) -> tuple:
        """Forward function.

        Args:
            inputs (tuple[Tensor]): Backbone outputs where the **last**
                element is the importance map ``(B, 1, H3, W3)`` and the
                preceding elements are the regular feature maps
                ``(C2, C3, C4, C5, ...)``.

        Returns:
            tuple[Tensor]: FPN output feature maps.
        """
        # Split off the importance map from the backbone outputs.
        imp_map = inputs[-1]          # (B, 1, H3, W3), values in [0, 1]
        inputs = inputs[:-1]          # (C2, C3, C4, C5)

        assert len(inputs) == len(self.in_channels)

        # Build laterals (same as standard FPN).
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # Build top-down path with importance-guided gating.
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # Upsample the higher-level feature to the spatial size of the
            # current lateral.
            target_size = laterals[i - 1].shape[2:]
            upsampled = F.interpolate(
                laterals[i], size=target_size, **self.upsample_cfg)

            # Resize the importance map to the same spatial size and apply
            # the learnable gate: (1 + alpha * imp_resized).
            imp_resized = F.interpolate(
                imp_map, size=target_size, mode='bilinear', align_corners=False)
            # Gate only the top-down upsampled signal; laterals[i-1] is
            # added as-is so the base FPN residual is always preserved.
            gate = 1.0 + self.alphas[i - 1] * imp_resized

            laterals[i - 1] = laterals[i - 1] + upsampled * gate

        # Build outputs — part 1: from original levels.
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        # Part 2: add extra levels (identical to standard FPN).
        if self.num_outs > len(outs):
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
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

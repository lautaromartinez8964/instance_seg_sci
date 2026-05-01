# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmdet.registry import MODELS

from .cspnext_pafpn import CSPNeXtPAFPN


@MODELS.register_module()
class DTCSPNeXtPAFPN(CSPNeXtPAFPN):
    """CSPNeXtPAFPN with distance-transform-guided top-down fusion.

    This is the RTMDet-era counterpart of DTFPN v2. It keeps the original
    CSPNeXtPAFPN topology, and only injects DT-guided gates into the top-down
    route so the migration stays local and easy to ablate.
    """

    def __init__(self,
                 in_channels: Sequence[int],
                 guided_levels: List[int] | None = None,
                 dt_mode: str = 'shared',
                 dt_decoder_source: str = 'inputs',
                 dt_head_channels: int = 64,
                 dt_loss_weight: float = 0.2,
                 skip_csp_on_guided: bool = False,
                 uagd_boundary_weight: float = 0.0,
                 uagd_boundary_channels: int = 32,
                 use_residual_fusion: bool = False,
                 use_channel_attention: bool = False,
                 **kwargs) -> None:
        super().__init__(in_channels=in_channels, **kwargs)
        # Keep backward compatibility with earlier config drafts. The current
        # RTMDet DT neck only uses gated top-down fusion and ignores these
        # legacy flags.
        self.use_residual_fusion = use_residual_fusion
        self.use_channel_attention = use_channel_attention
        num_fusions = len(self.in_channels) - 1
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
        if dt_mode not in {'shared', 'per_level', 'shared_refined'}:
            raise ValueError(
                f'Unsupported dt_mode={dt_mode}. Expected one of '
                "{'shared', 'per_level', 'shared_refined'}.")
        if dt_decoder_source not in {'inputs', 'topdown'}:
            raise ValueError(
                f'Unsupported dt_decoder_source={dt_decoder_source}. '
                "Expected one of {'inputs', 'topdown'}.")

        self.dt_mode = dt_mode
        self.dt_decoder_source = dt_decoder_source
        self.skip_csp_on_guided = skip_csp_on_guided
        if self.dt_decoder_source == 'inputs':
            dt_decoder_in_channels = list(self.in_channels)
        else:
            # _build_dt_decoder_inputs() reuses the CSPNeXt top-down route, so
            # its returned tuple channels become [c0] + in_channels[:-1].
            dt_decoder_in_channels = [self.in_channels[0], *self.in_channels[:-1]]

        if self.dt_mode in {'shared', 'shared_refined'}:
            self.dt_lateral_adapters = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(channels, dt_head_channels, kernel_size=3,
                              padding=1),
                    nn.ReLU(inplace=True))
                for channels in dt_decoder_in_channels
            ])
            self.dt_fuse_blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(dt_head_channels * 2, dt_head_channels,
                              kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(dt_head_channels, dt_head_channels,
                              kernel_size=3, padding=1),
                    nn.ReLU(inplace=True))
                for _ in range(num_fusions)
            ])
            self.dt_predictor = nn.Conv2d(dt_head_channels, 1, kernel_size=1)
        if self.dt_mode == 'per_level':
            self.per_level_dt_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(self.in_channels[level] * 2, dt_head_channels,
                              kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(dt_head_channels, 1, kernel_size=1))
                for level in range(num_fusions)
            ])
        elif self.dt_mode == 'shared_refined':
            self.per_level_dt_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(self.in_channels[level] * 2 + 1,
                              dt_head_channels,
                              kernel_size=3,
                              padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(dt_head_channels, dt_head_channels,
                              kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(dt_head_channels, 1, kernel_size=1))
                for level in range(num_fusions)
            ])

        self.gate_scales = nn.Parameter(torch.ones(num_fusions))
        self.gate_biases = nn.Parameter(torch.zeros(num_fusions))
        self._dt_loss_weight = float(max(dt_loss_weight, 0.0))

        self._current_dt_target: Tensor | None = None
        self._current_dt_valid_mask: Tensor | None = None
        self._last_dt_map: Tensor | None = None
        self._last_dt_maps: Tuple[Tensor, ...] = ()
        self._last_dt_loss: Tensor | None = None
        self.last_gates: Tuple[Tensor, ...] = ()

        # UAGD L1 boundary supervision head
        self._uagd_boundary_weight = float(max(uagd_boundary_weight, 0.0))
        self._current_b_tea_target: Tensor | None = None
        self._last_b_loss: Tensor | None = None
        if self._uagd_boundary_weight > 0:
            self.boundary_head = nn.Sequential(
                nn.Conv2d(self.out_channels, uagd_boundary_channels,
                          kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(uagd_boundary_channels, 1, kernel_size=1))

    def _extract_dt_target(self, data_sample,
                           device: torch.device) -> tuple[Tensor | None, Tensor | None]:
        if 'gt_sem_seg' not in data_sample:
            return None, None
        dt_target = data_sample.gt_sem_seg.sem_seg.to(
            device=device, dtype=torch.float32)
        if dt_target.ndim == 2:
            dt_target = dt_target.unsqueeze(0)
        elif dt_target.ndim == 4 and dt_target.shape[0] == 1:
            dt_target = dt_target.squeeze(0)
        if dt_target.ndim == 3 and dt_target.shape[-1] == 1 \
                and dt_target.shape[0] != 1:
            dt_target = dt_target.permute(2, 0, 1).contiguous()
        dt_target = dt_target[:1]
        valid_mask = torch.ones_like(dt_target, dtype=torch.float32)
        if dt_target.max() > 1.0:
            valid_mask = (dt_target < 254.5).to(dtype=torch.float32)
            dt_target = dt_target / 255.0
        dt_target = dt_target.clamp_(0.0, 1.0) * valid_mask
        return dt_target, valid_mask

    def set_auxiliary_targets(self,
                              batch_data_samples,
                              input_shape: Tuple[int, int] | None = None,
                              device=None) -> None:
        device = device or next(self.parameters()).device
        targets = []
        valid_masks = []
        for data_sample in batch_data_samples:
            dt_target, valid_mask = self._extract_dt_target(data_sample, device)
            if dt_target is None:
                self.clear_auxiliary_targets()
                return
            if input_shape is not None and dt_target.shape[-2:] != input_shape:
                pad_h = input_shape[0] - dt_target.shape[-2]
                pad_w = input_shape[1] - dt_target.shape[-1]
                if pad_h >= 0 and pad_w >= 0:
                    dt_target = F.pad(dt_target, (0, pad_w, 0, pad_h))
                    valid_mask = F.pad(valid_mask, (0, pad_w, 0, pad_h))
                else:
                    dt_target = F.interpolate(
                        dt_target.unsqueeze(0),
                        size=input_shape,
                        mode='bilinear',
                        align_corners=False).squeeze(0)
                    valid_mask = F.interpolate(
                        valid_mask.unsqueeze(0),
                        size=input_shape,
                        mode='nearest').squeeze(0)
            targets.append(dt_target)
            valid_masks.append(valid_mask)
        self._current_dt_target = torch.stack(targets, dim=0) if targets else None
        self._current_dt_valid_mask = torch.stack(valid_masks, dim=0) \
            if valid_masks else None
        self._last_dt_map = None
        self._last_dt_maps = ()
        self._last_dt_loss = None

        # Extract UAGD B_tea targets (same gt_sem_seg channel, ÷255 → [0,1])
        if self._uagd_boundary_weight > 0:
            b_targets = []
            for data_sample in batch_data_samples:
                if 'gt_sem_seg' not in data_sample:
                    self._current_b_tea_target = None
                    break
                b_tea = data_sample.gt_sem_seg.sem_seg.to(
                    device=device, dtype=torch.float32)
                if b_tea.ndim == 2:
                    b_tea = b_tea.unsqueeze(0)
                b_tea = b_tea[:1]
                b_tea = b_tea / 255.0  # uint8 0-255 → float [0, 1]
                b_tea = b_tea.clamp_(0.0, 1.0)
                if input_shape is not None and b_tea.shape[-2:] != input_shape:
                    pad_h = input_shape[0] - b_tea.shape[-2]
                    pad_w = input_shape[1] - b_tea.shape[-1]
                    if pad_h >= 0 and pad_w >= 0:
                        b_tea = F.pad(b_tea, (0, pad_w, 0, pad_h))
                    else:
                        b_tea = F.interpolate(
                            b_tea.unsqueeze(0), size=input_shape,
                            mode='bilinear', align_corners=False).squeeze(0)
                b_targets.append(b_tea)
            else:
                self._current_b_tea_target = \
                    torch.stack(b_targets, dim=0) if b_targets else None
        self._last_b_loss = None

    def clear_auxiliary_targets(self) -> None:
        self._current_dt_target = None
        self._current_dt_valid_mask = None
        self._last_dt_map = None
        self._last_dt_maps = ()
        self._last_dt_loss = None
        self.last_gates = ()
        self._current_b_tea_target = None
        self._last_b_loss = None

    def _compute_uagd_boundary_loss(self, b_stu: Tensor) -> Tensor | None:
        """MSE between predicted boundary map and B_tea target."""
        if self._current_b_tea_target is None \
                or self._uagd_boundary_weight <= 0:
            return None
        b_tea = self._current_b_tea_target  # (B, 1, H, W)
        # Downsample B_tea to P3 stride (1/8)
        target_size = b_stu.shape[-2:]
        b_tea_down = F.interpolate(
            b_tea, size=target_size, mode='bilinear', align_corners=False)
        loss = F.mse_loss(b_stu, b_tea_down)
        return self._uagd_boundary_weight * loss

    def _compute_dt_loss(self,
                         dt_maps: Tensor | Tuple[Tensor, ...]) -> Tensor | None:
        if self._current_dt_target is None or self._dt_loss_weight <= 0:
            return None
        if isinstance(dt_maps, Tensor):
            dt_maps = (dt_maps, )
        if not dt_maps:
            return None

        loss_terms = []
        for dt_map in dt_maps:
            dt_target = self._current_dt_target.to(
                device=dt_map.device, dtype=dt_map.dtype)
            valid_mask = self._current_dt_valid_mask
            if valid_mask is not None:
                valid_mask = valid_mask.to(device=dt_map.device,
                                           dtype=dt_map.dtype)
            if dt_target.shape[-2:] != dt_map.shape[-2:]:
                dt_target = F.interpolate(
                    dt_target,
                    size=dt_map.shape[-2:],
                    mode='bilinear',
                    align_corners=False)
                if valid_mask is not None:
                    valid_mask = F.interpolate(
                        valid_mask,
                        size=dt_map.shape[-2:],
                        mode='nearest')
            with torch.amp.autocast(device_type=dt_map.device.type,
                                    enabled=False):
                if valid_mask is not None:
                    squared_error = (dt_map.float() - dt_target.float())**2
                    valid_pixels = valid_mask.float() > 0.5
                    if valid_pixels.any():
                        loss_terms.append(squared_error[valid_pixels].mean())
                else:
                    loss_terms.append(F.mse_loss(dt_map.float(),
                                                 dt_target.float()))
        return torch.stack(loss_terms).mean() * self._dt_loss_weight

    def get_auxiliary_losses(self) -> dict:
        losses = {}
        if self._last_dt_loss is not None:
            losses['loss_neck_dt'] = self._last_dt_loss
        if self._last_b_loss is not None:
            losses['loss_neck_uagd_b'] = self._last_b_loss
        self.clear_auxiliary_targets()
        return losses

    def get_last_dt_map(self) -> Tensor | None:
        return self._last_dt_map

    def get_last_dt_maps(self) -> Tuple[Tensor, ...]:
        return self._last_dt_maps

    def _decode_shared_dt_map(self, inputs: Tuple[Tensor, ...]) -> Tensor:
        dt_feat = self.dt_lateral_adapters[-1](inputs[-1])
        for level in range(len(inputs) - 2, -1, -1):
            dt_feat = F.interpolate(
                dt_feat,
                size=inputs[level].shape[2:],
                mode='bilinear',
                align_corners=False)
            lateral_feat = self.dt_lateral_adapters[level](inputs[level])
            dt_feat = self.dt_fuse_blocks[level](
                torch.cat([dt_feat, lateral_feat], dim=1))
        return torch.sigmoid(self.dt_predictor(dt_feat))

    def _build_dt_decoder_inputs(self,
                                 inputs: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        if self.dt_decoder_source == 'inputs':
            return inputs

        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = inputs[idx - 1]
            feat_high = self.reduce_layers[len(self.in_channels) - 1 - idx](
                feat_high)
            inner_outs[0] = feat_high

            upsample_feat = self.upsample(feat_high)
            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](
                torch.cat([upsample_feat, feat_low], 1))
            inner_outs.insert(0, inner_out)

        return tuple(inner_outs)

    @staticmethod
    def _resize_dt_map(dt_map: Tensor, size: Tuple[int, int]) -> Tensor:
        if dt_map.shape[2:] == size:
            return dt_map
        return F.interpolate(
            dt_map, size=size, mode='bilinear', align_corners=False)

    def forward(self, inputs: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        assert len(inputs) == len(self.in_channels)

        dt_map = None
        if self.dt_mode in {'shared', 'shared_refined'}:
            dt_map = self._decode_shared_dt_map(
                self._build_dt_decoder_inputs(inputs))

        inner_outs = [inputs[-1]]
        gate_records: List[Tensor] = []
        per_level_dt_maps: List[Tensor | None] = [None] * len(self.gate_scales)

        for idx in range(len(self.in_channels) - 1, 0, -1):
            fusion_idx = len(self.in_channels) - 1 - idx
            feat_high = inner_outs[0]
            feat_low = inputs[idx - 1]
            feat_high = self.reduce_layers[fusion_idx](feat_high)
            inner_outs[0] = feat_high

            upsample_feat = self.upsample(feat_high)
            guided_low = feat_low

            target_level = idx - 1
            if target_level in self.guided_levels:
                if self.dt_mode == 'shared':
                    assert dt_map is not None
                    resized_dt = self._resize_dt_map(
                        dt_map, feat_low.shape[2:])
                elif self.dt_mode == 'per_level':
                    route_feat = torch.cat([feat_low, upsample_feat], dim=1)
                    resized_dt = torch.sigmoid(
                        self.per_level_dt_heads[target_level](route_feat))
                    per_level_dt_maps[target_level] = resized_dt
                else:
                    assert dt_map is not None
                    shared_resized_dt = self._resize_dt_map(
                        dt_map, feat_low.shape[2:])
                    route_feat = torch.cat([
                        feat_low, upsample_feat,
                        shared_resized_dt.to(dtype=upsample_feat.dtype)
                    ], dim=1)
                    resized_dt = torch.sigmoid(
                        self.per_level_dt_heads[target_level](route_feat))
                    per_level_dt_maps[target_level] = resized_dt

                scale = self.gate_scales[target_level].to(
                    dtype=upsample_feat.dtype).view(1, 1, 1, 1)
                bias = self.gate_biases[target_level].to(
                    dtype=upsample_feat.dtype).view(1, 1, 1, 1)
                gate = torch.sigmoid(
                    scale * resized_dt.to(dtype=upsample_feat.dtype) + bias)
                guided_low = feat_low * (1.0 - gate) + upsample_feat * gate
                gate_records.insert(0, gate.detach())

            if target_level in self.guided_levels and self.skip_csp_on_guided:
                inner_out = guided_low
            else:
                inner_out = self.top_down_blocks[fusion_idx](
                    torch.cat([upsample_feat, guided_low], 1))
            inner_outs.insert(0, inner_out)

        self.last_gates = tuple(gate_records)

        if self.dt_mode == 'shared':
            self._last_dt_map = dt_map
            self._last_dt_maps = ()
            self._last_dt_loss = self._compute_dt_loss(dt_map) \
                if dt_map is not None else None
        elif self.dt_mode == 'per_level':
            ordered_dt_maps = tuple(
                per_level_dt_maps[level] for level in self.guided_levels
                if per_level_dt_maps[level] is not None)
            self._last_dt_maps = ordered_dt_maps
            self._last_dt_map = ordered_dt_maps[0] if ordered_dt_maps else None
            self._last_dt_loss = self._compute_dt_loss(ordered_dt_maps)
        else:
            ordered_dt_maps = tuple(
                per_level_dt_maps[level] for level in self.guided_levels
                if per_level_dt_maps[level] is not None)
            self._last_dt_map = dt_map
            self._last_dt_maps = ordered_dt_maps
            loss_terms = []
            shared_loss = self._compute_dt_loss(dt_map) \
                if dt_map is not None else None
            refined_loss = self._compute_dt_loss(ordered_dt_maps)
            if shared_loss is not None:
                loss_terms.append(shared_loss)
            if refined_loss is not None:
                loss_terms.append(refined_loss)
            self._last_dt_loss = torch.stack(loss_terms).mean() \
                if loss_terms else None

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsamples[idx](feat_low)
            out = self.bottom_up_blocks[idx](
                torch.cat([downsample_feat, feat_high], 1))
            outs.append(out)

        for idx, conv in enumerate(self.out_convs):
            outs[idx] = conv(outs[idx])

        # UAGD L1: predict boundary map from P3 (outs[0], stride 8)
        if self._uagd_boundary_weight > 0 and self.training:
            b_stu = torch.sigmoid(self.boundary_head(outs[0]))  # (B,1,H/8,W/8)
            self._last_b_loss = self._compute_uagd_boundary_loss(b_stu)

        return tuple(outs)

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS

from mmdet.models.backbones.rs_lightmamba.lightmamba_backbone import \
    RSLightMambaBackbone


class InterBoundaryDetectionHead(nn.Module):
    """Fuse Stage2/Stage3 features to predict inter-instance boundaries."""

    def __init__(self, low_dim: int, high_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.low_proj = nn.Sequential(
            nn.Conv2d(low_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU())
        self.high_proj = nn.Sequential(
            nn.Conv2d(high_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU())
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU())
        self.predictor = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, low_feat: torch.Tensor,
                high_feat: torch.Tensor) -> torch.Tensor:
        high_feat = F.interpolate(
            high_feat,
            size=low_feat.shape[-2:],
            mode='bilinear',
            align_corners=False)
        fused = torch.cat([
            self.low_proj(low_feat),
            self.high_proj(high_feat)
        ], dim=1)
        return self.predictor(self.fuse(fused))


@MODELS.register_module()
class GravelLightMambaIBDBackbone(RSLightMambaBackbone):
    """RSLightMamba backbone with an IBD auxiliary branch for dense gravel."""

    def __init__(self,
                 ibd_stages: Sequence[int] = (1, 2),
                 ibd_hidden_dim: int = 128,
                 ibd_loss_weight: float = 1.0,
                 ibd_bce_weight: float = 1.0,
                 ibd_dice_weight: float = 1.0,
                 ibd_max_pos_weight: float = 8.0,
                 **kwargs) -> None:
        self._ibd_stages = tuple(ibd_stages)
        if len(self._ibd_stages) != 2:
            raise ValueError('ibd_stages must contain exactly two stage indices.')
        if self._ibd_stages[0] >= self._ibd_stages[1]:
            raise ValueError('ibd_stages must be ordered from low to high stage.')

        super().__init__(**kwargs)

        for stage_idx in self._ibd_stages:
            if stage_idx not in self.out_indices:
                raise ValueError('ibd_stages must be included in out_indices.')

        low_stage, high_stage = self._ibd_stages
        self.ibd_head = InterBoundaryDetectionHead(
            low_dim=self.dims[low_stage],
            high_dim=self.dims[high_stage],
            hidden_dim=ibd_hidden_dim)

        self._ibd_loss_weight = float(max(ibd_loss_weight, 0.0))
        self._ibd_bce_weight = float(max(ibd_bce_weight, 0.0))
        self._ibd_dice_weight = float(max(ibd_dice_weight, 0.0))
        self._ibd_max_pos_weight = float(max(ibd_max_pos_weight, 1.0))

        self._current_boundary_target: torch.Tensor | None = None
        self._last_boundary_logits: torch.Tensor | None = None
        self._last_boundary_loss: torch.Tensor | None = None

    def _build_boundary_target(self, batch_data_samples, input_shape,
                               device) -> torch.Tensor | None:
        if not batch_data_samples:
            return None

        target_h, target_w = input_shape
        batch_targets = []
        for data_sample in batch_data_samples:
            boundary_target = torch.zeros(
                (1, target_h, target_w), dtype=torch.float32, device=device)
            if 'gt_sem_seg' in data_sample:
                sem_seg = data_sample.gt_sem_seg.sem_seg.to(
                    device=device, dtype=torch.float32)
                if sem_seg.ndim == 2:
                    sem_seg = sem_seg.unsqueeze(0)
                sem_seg = (sem_seg > 0).float()
                if sem_seg.shape[-2:] != (target_h, target_w):
                    sem_seg = F.interpolate(
                        sem_seg.unsqueeze(0),
                        size=(target_h, target_w),
                        mode='nearest').squeeze(0)
                boundary_target = sem_seg
            batch_targets.append(boundary_target)
        return torch.stack(batch_targets, dim=0)

    def set_auxiliary_targets(self, batch_data_samples, input_shape,
                              device=None) -> None:
        device = device or next(self.parameters()).device
        self._current_boundary_target = self._build_boundary_target(
            batch_data_samples, input_shape, device)
        self._last_boundary_logits = None
        self._last_boundary_loss = None

    def clear_auxiliary_targets(self) -> None:
        self._current_boundary_target = None
        self._last_boundary_logits = None
        self._last_boundary_loss = None

    def _compute_ibd_loss(self, boundary_logits: torch.Tensor) -> torch.Tensor | None:
        if self._current_boundary_target is None or self._ibd_loss_weight <= 0:
            return None

        boundary_target = self._current_boundary_target.to(
            device=boundary_logits.device, dtype=boundary_logits.dtype)
        if boundary_target.shape[-2:] != boundary_logits.shape[-2:]:
            boundary_target = F.interpolate(
                boundary_target,
                size=boundary_logits.shape[-2:],
                mode='nearest')

        positive = boundary_target.sum()
        negative = boundary_target.numel() - positive
        pos_weight = (negative / positive.clamp_min(1.0)).clamp(
            min=1.0, max=self._ibd_max_pos_weight)
        pixel_weight = torch.where(
            boundary_target > 0.5,
            torch.full_like(boundary_target, pos_weight),
            torch.ones_like(boundary_target))

        with torch.amp.autocast(device_type=boundary_logits.device.type, enabled=False):
            bce = F.binary_cross_entropy_with_logits(
                boundary_logits.float(),
                boundary_target.float(),
                reduction='none')
            weighted_bce = (bce * pixel_weight.float()).sum() / pixel_weight.sum().clamp_min(1.0)

            boundary_prob = boundary_logits.float().sigmoid()
            intersection = (boundary_prob * boundary_target.float()).sum(dim=(1, 2, 3))
            union = boundary_prob.sum(dim=(1, 2, 3)) + boundary_target.float().sum(dim=(1, 2, 3))
            dice = 1.0 - (2.0 * intersection + 1.0) / (union + 1.0)

        loss = self._ibd_bce_weight * weighted_bce + self._ibd_dice_weight * dice.mean()
        return loss * self._ibd_loss_weight

    def get_auxiliary_losses(self) -> dict:
        loss = self._last_boundary_loss
        self.clear_auxiliary_targets()
        if loss is None:
            return {}
        return {'loss_ibd_boundary': loss}

    def get_last_boundary_logits(self) -> torch.Tensor | None:
        return self._last_boundary_logits

    def forward(self, x):
        def layer_forward(layer, hidden):
            hidden = layer.blocks(hidden)
            downsampled = layer.downsample(hidden)
            return hidden, downsampled

        x = self.patch_embed(x)
        outs = []
        hf_maps = []
        stage_outputs = {}
        for stage_idx, layer in enumerate(self.layers):
            stage_out, x = layer_forward(layer, x)
            if stage_idx in self.out_indices:
                norm_layer = getattr(self, f'outnorm{stage_idx}')
                out = norm_layer(stage_out)
                if not self.channel_first:
                    out = out.permute(0, 3, 1, 2)
                out = out.contiguous()
                stage_outputs[stage_idx] = out

                if (self._attention_importance_head is not None
                        and stage_idx == self._attention_fg_stage):
                    self._last_attention_importance = self._attention_importance_head(out)
                    self._last_attention_fg_loss = self._compute_attention_fg_loss(
                        self._last_attention_importance)

                if self._output_hf_maps and stage_idx in self._hf_map_stages:
                    hf_maps.append(self._hf_map_extractor(out))

                outs.append(out)

        if len(self.out_indices) == 0:
            return x

        low_stage, high_stage = self._ibd_stages
        self._last_boundary_logits = self.ibd_head(
            stage_outputs[low_stage], stage_outputs[high_stage])
        self._last_boundary_loss = self._compute_ibd_loss(self._last_boundary_logits)

        if self._output_guidance_map and self._guidance_head is not None:
            guide_low_stage, guide_high_stage = self._guidance_stages
            self._last_guidance_map = self._guidance_head(
                stage_outputs[guide_low_stage], stage_outputs[guide_high_stage])
            self._last_guidance_fg_loss = self._compute_guidance_fg_loss(
                self._last_guidance_map)
        if self._output_hf_maps:
            self._last_hf_maps = tuple(hf_maps)
            return tuple(outs), self._last_hf_maps
        if self._output_guidance_map:
            return tuple(outs), self._last_guidance_map
        return tuple(outs)
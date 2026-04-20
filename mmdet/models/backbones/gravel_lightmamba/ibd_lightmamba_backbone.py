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


class BoundaryConditionedResidualBlock(nn.Module):
    """Refine interior responses while suppressing cross-boundary mixing."""

    def __init__(self, dim: int, detach_boundary: bool = True) -> None:
        """
        初始化内部特征增强模块

        Args:
            dim (int): 输入特征通道数
            detach_boundary (bool, optional): 是否对边界信息进行梯度截断，默认为True
        """
        super().__init__()
        self.detach_boundary = detach_boundary
        self.interior_enhance = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim))
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, feat: torch.Tensor,
                boundary_prob: torch.Tensor) -> torch.Tensor:
        if self.detach_boundary:
            boundary_prob = boundary_prob.detach()
        if boundary_prob.shape[-2:] != feat.shape[-2:]:
            boundary_prob = F.interpolate(
                boundary_prob,
                size=feat.shape[-2:],
                mode='bilinear',
                align_corners=False)
        boundary_prob = boundary_prob.to(device=feat.device, dtype=feat.dtype).clamp_(0.0, 1.0)
        interior_mask = 1.0 - boundary_prob
        refined = feat + self.interior_enhance(feat) * interior_mask
        return feat + torch.tanh(self.alpha) * (refined - feat)


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
                 ibd_band_loss_weight: float = 0.35,
                 ibd_far_neg_weight: float = 2.0,
                 bcra_stages: Sequence[int] = (),
                 bcra_detach_boundary: bool = True,
                 **kwargs) -> None:
        self._ibd_stages = tuple(ibd_stages)
        if len(self._ibd_stages) != 2:
            raise ValueError('ibd_stages must contain exactly two stage indices.')
        if self._ibd_stages[0] >= self._ibd_stages[1]:
            raise ValueError('ibd_stages must be ordered from low to high stage.')
        self._bcra_stages = tuple(sorted(set(bcra_stages)))

        super().__init__(**kwargs)

        for stage_idx in self._ibd_stages:
            if stage_idx not in self.out_indices:
                raise ValueError('ibd_stages must be included in out_indices.')
        for stage_idx in self._bcra_stages:
            if stage_idx not in self.out_indices:
                raise ValueError('bcra_stages must be included in out_indices.')
            if stage_idx <= self._ibd_stages[1]:
                raise ValueError('bcra_stages must be deeper than the IBD high stage.')

        low_stage, high_stage = self._ibd_stages
        self.ibd_head = InterBoundaryDetectionHead(
            low_dim=self.dims[low_stage],
            high_dim=self.dims[high_stage],
            hidden_dim=ibd_hidden_dim)
        self.bcra_blocks = nn.ModuleDict({
            str(stage_idx): BoundaryConditionedResidualBlock(
                dim=self.dims[stage_idx],
                detach_boundary=bcra_detach_boundary)
            for stage_idx in self._bcra_stages
        })

        self._ibd_loss_weight = float(max(ibd_loss_weight, 0.0))
        self._ibd_bce_weight = float(max(ibd_bce_weight, 0.0))
        self._ibd_dice_weight = float(max(ibd_dice_weight, 0.0))
        self._ibd_max_pos_weight = float(max(ibd_max_pos_weight, 1.0))
        self._ibd_band_loss_weight = float(max(ibd_band_loss_weight, 0.0))
        self._ibd_far_neg_weight = float(max(ibd_far_neg_weight, 1.0))

        self._current_boundary_target: torch.Tensor | None = None
        self._last_boundary_logits: torch.Tensor | None = None
        self._last_boundary_loss: torch.Tensor | None = None

    def _build_boundary_target(self, batch_data_samples, input_shape,
                               device) -> dict[str, torch.Tensor] | None:
        if not batch_data_samples:
            return None

        target_h, target_w = input_shape
        core_targets = []
        band_targets = []
        for data_sample in batch_data_samples:
            core_target = torch.zeros(
                (1, target_h, target_w), dtype=torch.float32, device=device)
            band_target = torch.zeros(
                (1, target_h, target_w), dtype=torch.float32, device=device)
            if 'gt_sem_seg' in data_sample:
                sem_seg = data_sample.gt_sem_seg.sem_seg.to(
                    device=device, dtype=torch.float32)
                if sem_seg.ndim == 2:
                    sem_seg = sem_seg.unsqueeze(0)
                elif sem_seg.ndim == 4 and sem_seg.shape[0] == 1:
                    sem_seg = sem_seg.squeeze(0)
                if sem_seg.ndim == 3 and sem_seg.shape[-1] <= 4 and sem_seg.shape[0] != sem_seg.shape[-1]:
                    sem_seg = sem_seg.permute(2, 0, 1).contiguous()
                sem_seg = (sem_seg > 0).float()
                if sem_seg.shape[-2:] != (target_h, target_w):
                    sem_seg = F.interpolate(
                        sem_seg.unsqueeze(0),
                        size=(target_h, target_w),
                        mode='nearest').squeeze(0)
                if sem_seg.shape[0] >= 2:
                    core_target = sem_seg[0:1]
                    band_target = sem_seg[1:2]
                else:
                    core_target = sem_seg[0:1]
                    band_target = sem_seg[0:1]
            core_targets.append(core_target)
            band_targets.append(torch.maximum(band_target, core_target))
        return dict(
            core=torch.stack(core_targets, dim=0),
            band=torch.stack(band_targets, dim=0))

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

        core_target = self._current_boundary_target['core'].to(
            device=boundary_logits.device, dtype=boundary_logits.dtype)
        band_target = self._current_boundary_target['band'].to(
            device=boundary_logits.device, dtype=boundary_logits.dtype)
        loss_logits = boundary_logits
        if loss_logits.shape[-2:] != core_target.shape[-2:]:
            loss_logits = F.interpolate(
                loss_logits,
                size=core_target.shape[-2:],
                mode='bilinear',
                align_corners=False)

        core_target = core_target.clamp_(0.0, 1.0)
        band_target = torch.maximum(band_target.clamp_(0.0, 1.0), core_target)

        positive = core_target.sum()
        negative = core_target.numel() - positive
        pos_weight = (negative / positive.clamp_min(1.0)).clamp(
            min=1.0, max=self._ibd_max_pos_weight)

        with torch.amp.autocast(device_type=boundary_logits.device.type, enabled=False):
            core_bce = F.binary_cross_entropy_with_logits(
                loss_logits.float(),
                core_target.float(),
                pos_weight=pos_weight.float(),
                reduction='mean')

            boundary_prob = loss_logits.float().sigmoid()
            intersection = (boundary_prob * core_target.float()).sum(dim=(1, 2, 3))
            union = boundary_prob.sum(dim=(1, 2, 3)) + core_target.float().sum(dim=(1, 2, 3))
            dice = 1.0 - (2.0 * intersection + 1.0) / (union + 1.0)

            band_violation = (boundary_prob * (1.0 - band_target.float())).mean()
            far_neg_penalty = band_violation * self._ibd_far_neg_weight
            band_recall_loss = (core_target.float() * (1.0 - boundary_prob) * band_target.float()).mean()

        loss = (
            self._ibd_bce_weight * core_bce +
            self._ibd_dice_weight * dice.mean() +
            self._ibd_band_loss_weight * band_recall_loss +
            far_neg_penalty)
        return loss * self._ibd_loss_weight

    def get_auxiliary_losses(self) -> dict:
        loss = self._last_boundary_loss
        self.clear_auxiliary_targets()
        if loss is None:
            return {}
        return {'loss_ibd_boundary': loss}

    def get_last_boundary_logits(self) -> torch.Tensor | None:
        return self._last_boundary_logits

    def _maybe_compute_boundary(self, stage_outputs: dict[int, torch.Tensor]) -> None:
        low_stage, high_stage = self._ibd_stages
        if self._last_boundary_logits is not None:
            return
        if low_stage not in stage_outputs or high_stage not in stage_outputs:
            return
        self._last_boundary_logits = self.ibd_head(
            stage_outputs[low_stage], stage_outputs[high_stage])
        self._last_boundary_loss = self._compute_ibd_loss(self._last_boundary_logits)

    def forward(self, x):
        def layer_forward(layer, hidden):
            hidden = layer.blocks(hidden)
            downsampled = layer.downsample(hidden)
            return hidden, downsampled

        x = self.patch_embed(x)
        outs = []
        hf_maps = []
        stage_outputs = {}
        self._last_boundary_logits = None
        self._last_boundary_loss = None
        for stage_idx, layer in enumerate(self.layers):
            stage_out, x = layer_forward(layer, x)
            if stage_idx in self.out_indices:
                norm_layer = getattr(self, f'outnorm{stage_idx}')
                out = norm_layer(stage_out)
                if not self.channel_first:
                    out = out.permute(0, 3, 1, 2)
                out = out.contiguous()

                if stage_idx == self._ibd_stages[1]:
                    stage_outputs[stage_idx] = out
                    self._maybe_compute_boundary(stage_outputs)
                    out = stage_outputs[stage_idx]
                elif stage_idx in self._bcra_stages and self._last_boundary_logits is not None:
                    boundary_prob = self._last_boundary_logits.float().sigmoid()
                    out = self.bcra_blocks[str(stage_idx)](out, boundary_prob)

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

        self._maybe_compute_boundary(stage_outputs)

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
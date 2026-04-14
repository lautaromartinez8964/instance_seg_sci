from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS

from mmdet.models.backbones.vmamba_official.vmamba import Backbone_VSSM, SS2D

from .fg_ig_scan import ForegroundHead
from .global_attention import RSGlobalAttentionBlock
from .ig_ss2d import IGSS2D


class SpatialHighFrequencyExtractor(nn.Module):
    """Fixed high-pass spatial extractor used to build single-channel HF maps."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        kernel = torch.tensor([[0.0, -1.0, 0.0],
                               [-1.0, 4.0, -1.0],
                               [0.0, -1.0, 0.0]], dtype=torch.float32)
        self.register_buffer('laplacian_kernel', kernel.view(1, 1, 3, 3))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        weight = self.laplacian_kernel.expand(channels, 1, 3, 3)
        filtered = F.conv2d(x, weight, padding=1, groups=channels)
        energy = filtered.abs().mean(dim=1, keepdim=True)
        scale = energy.amax(dim=(2, 3), keepdim=True).clamp_min(self.eps)
        return energy / scale


class CrossStageGuidanceHead(nn.Module):
    """Fuse a lower-detail stage and a higher-semantic stage into one guidance map."""

    def __init__(self,
                 low_dim: int,
                 high_dim: int,
                 hidden_dim: int = 256,
                 lk_size: int = 7,
                 norm_type: str = 'bn',
                 gn_groups: int = 8):
        super().__init__()
        self.low_proj = nn.Conv2d(low_dim, hidden_dim, kernel_size=1, bias=False)
        self.high_proj = nn.Conv2d(high_dim, hidden_dim, kernel_size=1, bias=False)
        self.head = ForegroundHead(
            d_inner=hidden_dim * 2,
            lk_size=lk_size,
            norm_type=norm_type,
            gn_groups=gn_groups)

    def forward(self, low_feat: torch.Tensor,
                high_feat: torch.Tensor) -> torch.Tensor:
        high_feat = F.interpolate(
            high_feat,
            size=low_feat.shape[-2:],
            mode='bilinear',
            align_corners=False)
        low_embed = self.low_proj(low_feat)
        high_embed = self.high_proj(high_feat)
        fused = torch.cat([low_embed, high_embed], dim=1)
        return self.head(fused)


@MODELS.register_module()
class RSLightMambaBackbone(Backbone_VSSM):
    """Research variant backbone derived from official VMamba.

    Design goals:
    1) Keep official VMamba baseline untouched.
    2) Provide a clean research entry for IG-Scan / FGEB.
    3) Support robust loading from official pretrained checkpoints.
    """

    def __init__(
        self,
        out_indices=(0, 1, 2, 3),
        norm_layer='ln',
        official_pretrained=None,
        pretrained_key='model',
        strict_pretrained=False,
        ig_scan_stages: Optional[List[int]] = None,
        ig_region_size: int = 4,
        ig_region_score_mode: str = 'avg',
        ig_guidance_scale: float = 0.1,
        ig_lk_size: int = 7,
        ig_descending_only: bool = False,
        ig_mode: str = 'scan',
        ig_gate_scale: float = 0.1,
        ig_dt_scale: float = 0.03,
        ig_output_scale: float = 0.05,
        ig_gate_mode: str = 'positive',
        ig_fg_norm_type: str = 'bn',
        ig_fg_gn_groups: int = 8,
        ig_use_fg_loss: bool = False,
        ig_fg_loss_weight: float = 0.0,
        attention_stages: Optional[List[int]] = None,
        attention_num_heads: int = 8,
        attention_mlp_ratio: float = 4.0,
        attention_qkv_bias: bool = True,
        attention_attn_drop: float = 0.0,
        attention_proj_drop: float = 0.0,
        attention_use_pos_embed: bool = True,
        attention_fg_stage: Optional[int] = None,
        attention_use_fg_loss: bool = False,
        attention_fg_loss_weight: float = 0.0,
        attention_fg_lk_size: int = 7,
        attention_fg_norm_type: str = 'bn',
        attention_fg_gn_groups: int = 8,
        output_hf_maps: bool = False,
        hf_map_stages: Optional[List[int]] = None,
        output_guidance_map: bool = False,
        guidance_stages: Optional[List[int]] = None,
        guidance_hidden_dim: int = 256,
        guidance_use_fg_loss: bool = False,
        guidance_loss_weight: float = 0.0,
        guidance_lk_size: int = 7,
        guidance_norm_type: str = 'bn',
        guidance_gn_groups: int = 8,
        **kwargs,
    ):
        self._ig_scan_stages = sorted(set(ig_scan_stages or []))
        self._ig_region_size = ig_region_size
        self._ig_guidance_scale = ig_guidance_scale
        self._ig_lk_size = ig_lk_size
        self._ig_descending_only = ig_descending_only
        self._ig_fg_norm_type = ig_fg_norm_type
        self._ig_fg_gn_groups = ig_fg_gn_groups
        self._ig_use_fg_loss = ig_use_fg_loss
        self._ig_fg_loss_weight = ig_fg_loss_weight
        self._attention_stages = sorted(set(attention_stages or []))
        self._attention_num_heads = attention_num_heads
        self._attention_mlp_ratio = attention_mlp_ratio
        self._attention_qkv_bias = attention_qkv_bias
        self._attention_attn_drop = attention_attn_drop
        self._attention_proj_drop = attention_proj_drop
        self._attention_use_pos_embed = attention_use_pos_embed
        self._attention_fg_stage = attention_fg_stage
        self._attention_use_fg_loss = attention_use_fg_loss
        self._attention_fg_loss_weight = attention_fg_loss_weight
        self._attention_fg_lk_size = attention_fg_lk_size
        self._attention_fg_norm_type = attention_fg_norm_type
        self._attention_fg_gn_groups = attention_fg_gn_groups
        self._output_hf_maps = output_hf_maps
        self._hf_map_stages = sorted(set(hf_map_stages or ([0, 1, 2]
                                                            if output_hf_maps else [])))
        self._output_guidance_map = output_guidance_map
        self._guidance_stages = list(guidance_stages or ([2, 3]
                                                         if output_guidance_map else []))
        self._guidance_hidden_dim = guidance_hidden_dim
        self._guidance_use_fg_loss = guidance_use_fg_loss
        self._guidance_loss_weight = guidance_loss_weight
        self._guidance_lk_size = guidance_lk_size
        self._guidance_norm_type = guidance_norm_type
        self._guidance_gn_groups = guidance_gn_groups
        self._ss2d_defaults = dict(
            ssm_init=kwargs.get('ssm_init', 'v0'),
            forward_type=kwargs.get('forward_type', 'v2'),
        )
        self._ig_modules: List[IGSS2D] = []
        self._current_fg_target: torch.Tensor | None = None
        self._last_attention_importance: torch.Tensor | None = None
        self._last_attention_fg_loss: torch.Tensor | None = None
        self._attention_importance_head: ForegroundHead | None = None
        self._last_hf_maps: tuple[torch.Tensor, ...] = ()
        self._last_guidance_map: torch.Tensor | None = None
        self._last_guidance_fg_loss: torch.Tensor | None = None

        if ig_mode != 'scan':
            raise ValueError(
                'Only ig_mode="scan" is kept after cleanup. '
                'Use the v3b reorder path or the new Stage 4 attention path.')
        if ig_region_score_mode != 'avg':
            raise ValueError(
                'Only avg region scoring is kept after cleanup to preserve '
                'the v3b rollback line.')
        if any(stage in self._attention_stages for stage in self._ig_scan_stages):
            overlap = sorted(set(self._attention_stages) & set(self._ig_scan_stages))
            raise ValueError(f'attention_stages and ig_scan_stages cannot overlap: {overlap}')
        if self._attention_use_fg_loss and self._attention_fg_stage is None:
            if not self._attention_stages:
                raise ValueError('attention_use_fg_loss=True requires attention_stages.')
            self._attention_fg_stage = self._attention_stages[-1]
        if self._attention_fg_stage is not None:
            if self._attention_fg_stage not in out_indices:
                raise ValueError('attention_fg_stage must be included in out_indices.')
            if self._attention_fg_stage not in self._attention_stages:
                raise ValueError('attention_fg_stage must be one of attention_stages.')
        if self._output_hf_maps:
            for stage_idx in self._hf_map_stages:
                if stage_idx not in out_indices:
                    raise ValueError(
                        'hf_map_stages must be included in out_indices so the '
                        'same normalized backbone features can be reused.')
        if self._output_guidance_map:
            if len(self._guidance_stages) != 2:
                raise ValueError('guidance_stages must contain exactly two stages, e.g. [2, 3].')
            low_stage, high_stage = self._guidance_stages
            if low_stage >= high_stage:
                raise ValueError('guidance_stages must be ordered low->high, e.g. [2, 3].')
            for stage_idx in self._guidance_stages:
                if stage_idx not in out_indices:
                    raise ValueError('guidance_stages must be included in out_indices.')

        # Disable parent preload and use explicit research loader for clarity.
        super().__init__(
            out_indices=out_indices,
            pretrained=None,
            norm_layer=norm_layer,
            **kwargs,
        )

        if self._attention_stages:
            self._upgrade_to_attention_stages()
        if self._ig_scan_stages:
            self._upgrade_to_ig_scan()
        self._hf_map_extractor = SpatialHighFrequencyExtractor()
        self._guidance_head: CrossStageGuidanceHead | None = None
        if self._output_guidance_map:
            low_stage, high_stage = self._guidance_stages
            self._guidance_head = CrossStageGuidanceHead(
                low_dim=self.dims[low_stage],
                high_dim=self.dims[high_stage],
                hidden_dim=self._guidance_hidden_dim,
                lk_size=self._guidance_lk_size,
                norm_type=self._guidance_norm_type,
                gn_groups=self._guidance_gn_groups)
        if self._attention_fg_stage is not None:
            self._attention_importance_head = ForegroundHead(
                d_inner=self.dims[self._attention_fg_stage],
                lk_size=self._attention_fg_lk_size,
                norm_type=self._attention_fg_norm_type,
                gn_groups=self._attention_fg_gn_groups)

        self.pretrained_key = pretrained_key
        self.strict_pretrained = strict_pretrained

        if official_pretrained:
            self.load_official_pretrained(
                official_pretrained,
                key=pretrained_key,
                strict=strict_pretrained,
            )

    def _build_igss2d_from_ss2d(self, old_op: SS2D,
                                block_idx: int) -> IGSS2D:
        conv_bias = False
        d_conv = 1
        if getattr(old_op, 'with_dconv', False):
            d_conv = int(old_op.conv2d.kernel_size[0])
            conv_bias = old_op.conv2d.bias is not None

        dropout = old_op.dropout.p if isinstance(old_op.dropout,
                                                 nn.Dropout) else 0.0
        bias = old_op.in_proj.bias is not None

        new_op = IGSS2D(
            d_model=old_op.d_model,
            d_state=old_op.d_state,
            ssm_ratio=(old_op.d_inner / old_op.d_model),
            dt_rank=old_op.dt_rank,
            act_layer=type(old_op.act),
            d_conv=d_conv,
            conv_bias=conv_bias,
            dropout=dropout,
            bias=bias,
            initialize=self._ss2d_defaults['ssm_init'],
            forward_type=self._ss2d_defaults['forward_type'],
            channel_first=old_op.channel_first,
            region_size=self._ig_region_size,
            guidance_scale=self._ig_guidance_scale,
            lk_size=self._ig_lk_size,
            fg_loss_weight=(self._ig_fg_loss_weight if self._ig_use_fg_loss else 0.0),
            fg_norm_type=self._ig_fg_norm_type,
            fg_gn_groups=self._ig_fg_gn_groups,
            descending_only=self._ig_descending_only,
            block_idx=block_idx,
        )

        old_state = old_op.state_dict()
        new_state = new_op.state_dict()
        compatible = {
            key: value
            for key, value in old_state.items()
            if key in new_state and new_state[key].shape == value.shape
        }
        new_op.load_state_dict(compatible, strict=False)
        return new_op

    def _build_attention_block(self, stage_idx: int,
                               old_block: nn.Module) -> RSGlobalAttentionBlock:
        drop_path = float(getattr(getattr(old_block, 'drop_path', None),
                                  'drop_prob', 0.0) or 0.0)
        return RSGlobalAttentionBlock(
            hidden_dim=self.dims[stage_idx],
            num_heads=self._attention_num_heads,
            mlp_ratio=self._attention_mlp_ratio,
            drop_path=drop_path,
            qkv_bias=self._attention_qkv_bias,
            attn_drop=self._attention_attn_drop,
            proj_drop=self._attention_proj_drop,
            use_checkpoint=getattr(old_block, 'use_checkpoint', False),
            channel_first=self.channel_first,
            use_pos_embed=self._attention_use_pos_embed)

    def _upgrade_to_attention_stages(self):
        replaced = 0
        for stage_idx in self._attention_stages:
            if stage_idx >= len(self.layers):
                print(f'[Stage4-Attn] Warning: stage {stage_idx} does not exist.')
                continue

            blocks = self.layers[stage_idx].blocks
            new_blocks = []
            for block in blocks:
                new_blocks.append(self._build_attention_block(stage_idx, block))
                replaced += 1
            self.layers[stage_idx].blocks = nn.Sequential(*new_blocks)

        print(
            f'[Stage4-Attn] Replaced {replaced} VSS blocks in stages '
            f'{self._attention_stages} with global attention.')

    def _upgrade_to_ig_scan(self):
        upgraded = 0
        self._ig_modules = []
        for stage_idx in self._ig_scan_stages:
            if stage_idx >= len(self.layers):
                print(f'[IG-Scan] Warning: stage {stage_idx} does not exist.')
                continue

            blocks = self.layers[stage_idx].blocks
            for block_idx, block in enumerate(blocks):
                if not hasattr(block, 'op') or not isinstance(block.op, SS2D):
                    continue
                block.op = self._build_igss2d_from_ss2d(block.op, block_idx)
                self._ig_modules.append(block.op)
                upgraded += 1

        print(
            f'[IG-Scan] Upgraded {upgraded} VSS blocks in stages '
            f'{self._ig_scan_stages} with mode=scan, '
            f'region_size={self._ig_region_size}.')

    @staticmethod
    def _copy_overlap(src_tensor: torch.Tensor,
                      dst_tensor: torch.Tensor) -> torch.Tensor:
        """Copy overlapped values for shape-mismatched tensors.

        - Channel-like dims: copy from start.
        - Spatial kernel dims: center-crop/pad alignment.
        """
        out = dst_tensor.clone()
        out.zero_()

        src_slices = []
        dst_slices = []
        for dim_idx, (src_dim, dst_dim) in enumerate(
                zip(src_tensor.shape, dst_tensor.shape)):
            copy_dim = min(src_dim, dst_dim)
            if src_tensor.ndim >= 4 and dim_idx >= src_tensor.ndim - 2:
                src_start = (src_dim - copy_dim) // 2
                dst_start = (dst_dim - copy_dim) // 2
            else:
                src_start = 0
                dst_start = 0
            src_slices.append(slice(src_start, src_start + copy_dim))
            dst_slices.append(slice(dst_start, dst_start + copy_dim))

        out[tuple(dst_slices)] = src_tensor[tuple(src_slices)].to(out.dtype)
        return out

    def _extract_state_dict(self, ckpt_path: str, key: str = 'model'):
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if isinstance(checkpoint, dict):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state_dict = checkpoint[key]
            elif 'state_dict' in checkpoint and isinstance(
                    checkpoint['state_dict'], dict):
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        filtered = {
            k: v
            for k, v in state_dict.items()
            if not k.startswith('classifier.') and not k.startswith('head')
        }
        return filtered

    @staticmethod
    def _expand_in_proj_tensor(src_tensor: torch.Tensor,
                               dst_tensor: torch.Tensor) -> torch.Tensor | None:
        if src_tensor.ndim == 2:
            if dst_tensor.shape[0] != src_tensor.shape[0] * 2 or dst_tensor.shape[1] != src_tensor.shape[1]:
                return None
            return torch.cat([src_tensor, src_tensor], dim=0).to(dst_tensor.dtype)
        if src_tensor.ndim == 1:
            if dst_tensor.shape[0] != src_tensor.shape[0] * 2:
                return None
            return torch.cat([src_tensor, src_tensor], dim=0).to(dst_tensor.dtype)
        return None

    def load_official_pretrained(self,
                                 ckpt_path: str,
                                 key: str = 'model',
                                 strict: bool = False):
        """Load official VMamba checkpoint with robust partial matching."""
        src_state = self._extract_state_dict(ckpt_path, key=key)
        dst_state = self.state_dict()

        merged = {}
        exact_params = 0
        partial_params = 0
        skipped = []

        for k, dst_v in dst_state.items():
            if k not in src_state:
                continue
            src_v = src_state[k]
            if k.endswith('in_proj.weight') or k.endswith('in_proj.bias'):
                expanded = self._expand_in_proj_tensor(src_v, dst_v)
                if expanded is not None:
                    merged[k] = expanded
                    exact_params += dst_v.numel()
                    continue
            if src_v.shape == dst_v.shape:
                merged[k] = src_v.to(dst_v.dtype)
                exact_params += dst_v.numel()
            elif src_v.ndim == dst_v.ndim:
                merged[k] = self._copy_overlap(src_v, dst_v)
                partial_params += min(src_v.numel(), dst_v.numel())
            else:
                skipped.append(k)

        msg = self.load_state_dict(merged, strict=False if not strict else True)
        total_params = sum(v.numel() for v in dst_state.values())
        loaded_params = exact_params + partial_params

        print(f'Loaded official checkpoint: {ckpt_path}')
        print(
            f'Loaded params: {loaded_params / 1e6:.3f}M / {total_params / 1e6:.3f}M '
            f'({loaded_params / total_params * 100:.2f}%)')
        print(
            f'Exact params: {exact_params / 1e6:.3f}M, '
            f'Partial params: {partial_params / 1e6:.3f}M, '
            f'Skipped keys: {len(skipped)}')
        print(f'Missing keys: {len(msg.missing_keys)}')
        print(f'Unexpected keys: {len(msg.unexpected_keys)}')

    def _build_fg_target(self, batch_data_samples, input_shape, device):
        if not batch_data_samples:
            return None

        batch_fg = []
        target_h, target_w = input_shape
        for data_sample in batch_data_samples:
            fg_map = torch.zeros((target_h, target_w), dtype=torch.float32, device=device)
            gt_instances = getattr(data_sample, 'gt_instances', None)
            if gt_instances is not None and hasattr(gt_instances, 'masks') and gt_instances.masks is not None:
                mask_tensor = gt_instances.masks.to_tensor(torch.float32, device)
                if mask_tensor.numel() > 0:
                    merged = mask_tensor.amax(dim=0)
                    mask_h = min(merged.shape[-2], target_h)
                    mask_w = min(merged.shape[-1], target_w)
                    fg_map[:mask_h, :mask_w] = merged[:mask_h, :mask_w]
            batch_fg.append(fg_map.unsqueeze(0))

        return torch.stack(batch_fg, dim=0)

    def set_ig_targets(self, batch_data_samples, input_shape, device=None) -> None:
        if (not self._ig_modules and not self._attention_use_fg_loss
                and not self._guidance_use_fg_loss):
            return

        device = device or next(self.parameters()).device
        fg_target = None
        if self._ig_use_fg_loss or self._attention_use_fg_loss or self._guidance_use_fg_loss:
            fg_target = self._build_fg_target(batch_data_samples, input_shape, device)
        self._current_fg_target = fg_target
        self._last_attention_importance = None
        self._last_attention_fg_loss = None
        self._last_guidance_map = None
        self._last_guidance_fg_loss = None

        for module in self._ig_modules:
            module.reset_runtime_state()
            module.set_fg_target(fg_target)

    def clear_ig_targets(self) -> None:
        self._current_fg_target = None
        self._last_attention_importance = None
        self._last_attention_fg_loss = None
        self._last_hf_maps = ()
        self._last_guidance_map = None
        self._last_guidance_fg_loss = None
        for module in self._ig_modules:
            module.clear_fg_target()
            module.reset_runtime_state()

    def set_ig_fg_loss_weight(self, fg_loss_weight: float) -> None:
        safe_weight = float(max(fg_loss_weight, 0.0))
        self._ig_fg_loss_weight = safe_weight
        if self._attention_use_fg_loss:
            self._attention_fg_loss_weight = safe_weight
        if self._guidance_use_fg_loss:
            self._guidance_loss_weight = safe_weight
        for module in self._ig_modules:
            module.set_fg_loss_weight(safe_weight)

    def _compute_attention_fg_loss(self,
                                   importance: torch.Tensor) -> torch.Tensor | None:
        if self._current_fg_target is None or not self._attention_use_fg_loss:
            return None
        return self._compute_shared_fg_loss(importance, self._attention_fg_loss_weight)

    def _compute_guidance_fg_loss(self,
                                  guidance_map: torch.Tensor) -> torch.Tensor | None:
        if self._current_fg_target is None or not self._guidance_use_fg_loss:
            return None
        return self._compute_shared_fg_loss(guidance_map, self._guidance_loss_weight)

    def _compute_shared_fg_loss(self,
                                pred_map: torch.Tensor,
                                loss_weight: float) -> torch.Tensor | None:
        if self._current_fg_target is None or loss_weight <= 0:
            return None

        fg_target = self._current_fg_target.to(
            device=pred_map.device, dtype=pred_map.dtype)
        if fg_target.shape[-2:] != pred_map.shape[-2:]:
            fg_target = torch.nn.functional.interpolate(
                fg_target,
                size=pred_map.shape[-2:],
                mode='area')
            fg_target = fg_target.clamp(0.0, 1.0)

        with torch.amp.autocast(device_type=pred_map.device.type, enabled=False):
            loss = torch.nn.functional.binary_cross_entropy(
                pred_map.float(),
                fg_target.float())
        return loss * loss_weight

    def get_ig_aux_losses(self) -> dict:
        loss_values = []
        if self._ig_use_fg_loss:
            for module in self._ig_modules:
                fg_loss = module.get_fg_loss()
                if fg_loss is not None:
                    loss_values.append(fg_loss)

        if self._last_attention_fg_loss is not None:
            loss_values.append(self._last_attention_fg_loss)
        if self._last_guidance_fg_loss is not None:
            loss_values.append(self._last_guidance_fg_loss)

        self.clear_ig_targets()
        if not loss_values:
            return {}

        return {'loss_fg': torch.stack(loss_values).mean()}

    def get_last_hf_maps(self) -> tuple[torch.Tensor, ...]:
        return self._last_hf_maps

    def get_last_guidance_map(self) -> torch.Tensor | None:
        return self._last_guidance_map

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
        if self._output_guidance_map and self._guidance_head is not None:
            low_stage, high_stage = self._guidance_stages
            self._last_guidance_map = self._guidance_head(
                stage_outputs[low_stage], stage_outputs[high_stage])
            self._last_guidance_fg_loss = self._compute_guidance_fg_loss(
                self._last_guidance_map)
        if self._output_hf_maps:
            self._last_hf_maps = tuple(hf_maps)
            return tuple(outs), self._last_hf_maps
        if self._output_guidance_map:
            return tuple(outs), self._last_guidance_map
        return tuple(outs)

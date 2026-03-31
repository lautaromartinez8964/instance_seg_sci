from typing import List, Optional

import torch
import torch.nn as nn
from mmdet.registry import MODELS

from mmdet.models.backbones.vmamba_official.vmamba import Backbone_VSSM, SS2D

from .ig_ss2d import IGSS2D


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
        ig_guidance_scale: float = 0.1,
        ig_lk_size: int = 7,
        ig_descending_only: bool = False,
        ig_use_fg_loss: bool = False,
        ig_fg_loss_weight: float = 0.0,
        **kwargs,
    ):
        self._ig_scan_stages = sorted(set(ig_scan_stages or []))
        self._ig_region_size = ig_region_size
        self._ig_guidance_scale = ig_guidance_scale
        self._ig_lk_size = ig_lk_size
        self._ig_descending_only = ig_descending_only
        self._ig_use_fg_loss = ig_use_fg_loss
        self._ig_fg_loss_weight = ig_fg_loss_weight
        self._ss2d_defaults = dict(
            ssm_init=kwargs.get('ssm_init', 'v0'),
            forward_type=kwargs.get('forward_type', 'v2'),
        )
        self._ig_modules: List[IGSS2D] = []

        # Disable parent preload and use explicit research loader for clarity.
        super().__init__(
            out_indices=out_indices,
            pretrained=None,
            norm_layer=norm_layer,
            **kwargs,
        )

        if self._ig_scan_stages:
            self._upgrade_to_ig_scan()

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
            f'{self._ig_scan_stages} with region_size={self._ig_region_size}.')

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
        if not self._ig_modules:
            return

        device = device or next(self.parameters()).device
        fg_target = None
        if self._ig_use_fg_loss:
            fg_target = self._build_fg_target(batch_data_samples, input_shape, device)

        for module in self._ig_modules:
            module.reset_runtime_state()
            module.set_fg_target(fg_target)

    def clear_ig_targets(self) -> None:
        for module in self._ig_modules:
            module.clear_fg_target()
            module.reset_runtime_state()

    def get_ig_aux_losses(self) -> dict:
        if not self._ig_modules or not self._ig_use_fg_loss:
            return {}

        loss_values = []
        for module in self._ig_modules:
            fg_loss = module.get_fg_loss()
            if fg_loss is not None:
                loss_values.append(fg_loss)

        self.clear_ig_targets()
        if not loss_values:
            return {}

        return {'loss_fg': torch.stack(loss_values).mean()}

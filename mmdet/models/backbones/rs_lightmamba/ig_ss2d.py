from __future__ import annotations

import torch

from mmdet.models.backbones.vmamba_official.vmamba import SS2D

from .fg_ig_scan import FGIGScan


class IGSS2D(SS2D):
    """SS2D with 2D-level foreground-guided region reordering."""

    def __init__(self,
                 region_size: int = 4,
                 block_idx: int = 0,
                 guidance_scale: float = 0.5,
                 **kwargs):
        super().__init__(**kwargs)
        self.ig_scan = FGIGScan(
            d_inner=self.d_inner,
            region_size=region_size,
            guidance_scale=guidance_scale,
        )
        self.block_idx = block_idx

    def forwardv2(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
            if not self.disable_z_act:
                z = self.act(z)
        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.with_dconv:
            x = self.conv2d(x)
        x = self.act(x)

        descending = (self.block_idx % 2 == 0)
        x = self.ig_scan.permute_2d(x, descending=descending)
        y = self.forward_core(x)

        if y.ndim == 4:
            if self.channel_first:
                y_cf = y
            else:
                y_cf = y.permute(0, 3, 1, 2).contiguous()
        elif y.ndim == 3:
            batch_size, _, height, width = x.shape
            if self.channel_first:
                y_cf = y.view(batch_size, -1, height, width)
            else:
                y_cf = y.view(batch_size, height, width, -1).permute(
                    0, 3, 1, 2).contiguous()
        else:
            raise RuntimeError(f'Unexpected forward_core output shape: {y.shape}')

        y_cf = self.ig_scan.inv_permute_2d(y_cf)

        if self.channel_first:
            y = y_cf
        else:
            y = y_cf.permute(0, 2, 3, 1).contiguous()

        y = self.out_act(y)
        if not self.disable_z:
            y = y * z
        return self.dropout(self.out_proj(y))
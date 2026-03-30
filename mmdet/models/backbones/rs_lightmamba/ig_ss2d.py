from __future__ import annotations

from functools import partial

import torch

from mmdet.models.backbones.vmamba_official.vmamba import SS2D
from mmdet.models.backbones.vmamba_official.csms6s import selective_scan_fn

from .fg_ig_scan import FGIGScan
from .ig_cross_scan import IGCrossScan


class IGSS2D(SS2D):
    """SS2D with region-priority 1D cross scan."""

    def __init__(self,
                 region_size: int = 4,
                 block_idx: int = 0,
                 guidance_scale: float = 0.1,
                 lk_size: int = 7,
                 descending_only: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        original_forward_core = self.forward_core
        self.ig_scan_module = FGIGScan(
            d_inner=self.d_inner,
            region_size=region_size,
            guidance_scale=guidance_scale,
            lk_size=lk_size,
        )
        self.ig_cross = IGCrossScan()
        self.block_idx = block_idx
        self.descending_only = descending_only
        core_keywords = dict(getattr(original_forward_core, 'keywords', {}) or {})
        core_keywords.setdefault('force_fp32', (not self.disable_force32))
        core_keywords.setdefault('selective_scan_backend', 'mamba')
        self.forward_core = partial(
            self.forward_core_ig,
            **core_keywords,
        )

    def forward_core_ig(self,
                        x: torch.Tensor = None,
                        force_fp32: bool = False,
                        ssoflex: bool = True,
                        no_einsum: bool = False,
                        selective_scan_backend: str = 'mamba',
                        **kwargs):
        del kwargs
        x_proj_bias = getattr(self, 'x_proj_bias', None)
        out_norm = self.out_norm
        channel_first = self.channel_first

        batch_size, dim, height, width = x.shape
        state_dim = self.d_state
        k_group = self.k_group
        dt_rank = self.dt_rank
        seq_len = height * width

        descending = True if self.descending_only else (self.block_idx % 2 == 0)
        x_mod, order, gh, gw, _ = self.ig_scan_module.compute_region_order(
            x, descending=descending)
        xs = self.ig_cross.scan(x_mod, order, gh, gw)

        if no_einsum:
            x_dbl = torch.nn.functional.conv1d(
                xs.view(batch_size, -1, seq_len),
                self.x_proj_weight.view(-1, dim, 1),
                bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None),
                groups=k_group)
            dts, Bs, Cs = torch.split(
                x_dbl.view(batch_size, k_group, -1, seq_len),
                [dt_rank, state_dim, state_dim],
                dim=2)
            if hasattr(self, 'dt_projs_weight'):
                dts = torch.nn.functional.conv1d(
                    dts.contiguous().view(batch_size, -1, seq_len),
                    self.dt_projs_weight.view(k_group * dim, -1, 1),
                    groups=k_group)
        else:
            x_dbl = torch.einsum('b k d l, k c d -> b k c l', xs,
                                 self.x_proj_weight)
            if x_proj_bias is not None:
                x_dbl = x_dbl + x_proj_bias.view(1, k_group, -1, 1)
            dts, Bs, Cs = torch.split(x_dbl, [dt_rank, state_dim, state_dim], dim=2)
            if hasattr(self, 'dt_projs_weight'):
                dts = torch.einsum('b k r l, k d r -> b k d l', dts,
                                   self.dt_projs_weight)

        xs = xs.view(batch_size, -1, seq_len)
        dts = dts.contiguous().view(batch_size, -1, seq_len)
        As = -self.A_logs.to(torch.float).exp()
        Ds = self.Ds.to(torch.float)
        Bs = Bs.contiguous().view(batch_size, k_group, state_dim, seq_len)
        Cs = Cs.contiguous().view(batch_size, k_group, state_dim, seq_len)
        delta_bias = self.dt_projs_bias.view(-1).to(torch.float)

        if force_fp32:
            xs = xs.to(torch.float32)
            dts = dts.to(torch.float32)
            Bs = Bs.to(torch.float32)
            Cs = Cs.to(torch.float32)

        ys = selective_scan_fn(
            xs, dts, As, Bs, Cs, Ds, delta_bias, True, ssoflex,
            backend=selective_scan_backend).view(batch_size, k_group, -1, seq_len)

        y = self.ig_cross.merge(ys).view(batch_size, -1, height, width)
        if not channel_first:
            y = y.view(batch_size, -1, seq_len).transpose(1, 2).contiguous().view(
                batch_size, height, width, -1)
        y = out_norm(y)
        return y.to(x.dtype)

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
        y = self.forward_core(x)
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z
        return self.dropout(self.out_proj(y))
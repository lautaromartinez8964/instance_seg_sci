from __future__ import annotations

from functools import partial
import math

import torch
import torch.nn.functional as F

from mmdet.models.backbones.vmamba_official.vmamba import SS2D, cross_merge_fn, cross_scan_fn
from mmdet.models.backbones.vmamba_official.csms6s import selective_scan_fn

from .fg_ig_scan import FGIGScan
from .ig_cross_scan import IGCrossScan


def _inverse_softplus(value: float) -> float:
    value = float(max(value, 1e-6))
    return value + math.log(-math.expm1(-value))


def apply_importance_to_z(z: torch.Tensor,
                          importance: torch.Tensor,
                          gate_scale: torch.Tensor,
                          gate_mode: str = 'positive') -> torch.Tensor:
    if gate_mode == 'positive':
        return z * (1.0 + torch.clamp(gate_scale, min=0.0) * importance)
    if gate_mode == 'bidirectional':
        centered_importance = importance.mul(2.0).sub(1.0)
        safe_gate_scale = 0.5 * torch.tanh(gate_scale)
        return z * (1.0 + safe_gate_scale * centered_importance)
    raise ValueError(f'Unsupported gate_mode: {gate_mode}')


def apply_importance_to_dt(dts: torch.Tensor,
                           importance: torch.Tensor,
                           dt_scale: torch.Tensor,
                           dt_bias: torch.Tensor) -> torch.Tensor:
    safe_dt_scale = dt_scale.to(dtype=dts.dtype)
    effective_dt = F.softplus(dts + dt_bias.to(dtype=dts.dtype))
    dt_factor = 1.0 + safe_dt_scale * importance.to(dtype=dts.dtype)
    return effective_dt * dt_factor


class IGSS2D(SS2D):
    """SS2D with region-priority 1D cross scan."""

    def __init__(self,
                 region_size: int = 4,
                 block_idx: int = 0,
                 guidance_scale: float = 0.1,
                 lk_size: int = 7,
                 fg_loss_weight: float = 0.0,
                 fg_norm_type: str = 'bn',
                 fg_gn_groups: int = 8,
                 ig_mode: str = 'scan',
                 gate_scale: float = 0.1,
                 dt_scale: float = 0.03,
                 gate_mode: str = 'positive',
                 descending_only: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        original_forward_core = self.forward_core
        self.ig_scan_module = FGIGScan(
            d_inner=self.d_inner,
            region_size=region_size,
            guidance_scale=guidance_scale,
            lk_size=lk_size,
            fg_loss_weight=fg_loss_weight,
            fg_norm_type=fg_norm_type,
            fg_gn_groups=fg_gn_groups,
        )
        self.ig_cross = IGCrossScan()
        self.block_idx = block_idx
        self.descending_only = descending_only
        self.ig_mode = ig_mode
        self.gate_mode = gate_mode
        self.gate_scale = torch.nn.Parameter(
            torch.tensor(float(max(gate_scale, 0.0)), dtype=torch.float32))
        self.dt_scale_raw = torch.nn.Parameter(
            torch.tensor(_inverse_softplus(max(dt_scale, 1e-6)),
                         dtype=torch.float32))
        self._official_forward_core = original_forward_core

        if self.ig_mode not in {'scan', 'z_gate', 'dt_gate'}:
            raise ValueError(f'Unsupported ig_mode: {self.ig_mode}')
        if self.gate_mode not in {'positive', 'bidirectional'}:
            raise ValueError(f'Unsupported gate_mode: {self.gate_mode}')
        if self.ig_mode == 'z_gate' and self.disable_z:
            raise ValueError('ig_mode="z_gate" requires z gating, but forward_type disables z.')

        if self.ig_mode in {'scan', 'dt_gate'}:
            core_keywords = dict(getattr(original_forward_core, 'keywords', {}) or {})
            core_keywords.setdefault('force_fp32', (not self.disable_force32))
            core_keywords.setdefault('selective_scan_backend', 'mamba')
            self.forward_core = partial(
                self.forward_core_ig,
                **core_keywords,
            )

    @property
    def dt_scale(self) -> torch.Tensor:
        return F.softplus(self.dt_scale_raw)

    def set_fg_target(self, fg_target: torch.Tensor | None) -> None:
        self.ig_scan_module.set_fg_target(fg_target)

    def clear_fg_target(self) -> None:
        self.ig_scan_module.clear_fg_target()

    def get_fg_loss(self) -> torch.Tensor | None:
        return self.ig_scan_module.last_fg_loss

    def reset_runtime_state(self) -> None:
        self.ig_scan_module.reset_state()

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

        if self.ig_mode == 'scan':
            descending = True if self.descending_only else (self.block_idx % 2 == 0)
            x_mod, order, gh, gw, _ = self.ig_scan_module.compute_region_order(
                x, descending=descending)
            xs = self.ig_cross.scan(x_mod, order, gh, gw)
        else:
            xs = cross_scan_fn(
                x,
                in_channel_first=True,
                out_channel_first=True,
                scans=0,
                force_torch=False)

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

        dts = dts.contiguous().view(batch_size, k_group, -1, seq_len)
        delta_bias = self.dt_projs_bias.view(1, k_group, -1, 1)
        delta_softplus = True
        if self.ig_mode == 'dt_gate':
            importance = self.ig_scan_module.predict_importance(x)
            importance_seq = cross_scan_fn(
                importance.to(dtype=dts.dtype),
                in_channel_first=True,
                out_channel_first=True,
                scans=0,
                force_torch=False).view(batch_size, k_group, 1, seq_len)
            dts = apply_importance_to_dt(
                dts,
                importance_seq,
                self.dt_scale,
                delta_bias)
            delta_bias = None
            delta_softplus = False

        xs = xs.view(batch_size, -1, seq_len)
        dts = dts.contiguous().view(batch_size, -1, seq_len)
        As = -self.A_logs.to(torch.float).exp()
        Ds = self.Ds.to(torch.float)
        Bs = Bs.contiguous().view(batch_size, k_group, state_dim, seq_len)
        Cs = Cs.contiguous().view(batch_size, k_group, state_dim, seq_len)
        if delta_bias is not None:
            delta_bias = delta_bias.view(-1).to(torch.float)

        if force_fp32:
            xs = xs.to(torch.float32)
            dts = dts.to(torch.float32)
            Bs = Bs.to(torch.float32)
            Cs = Cs.to(torch.float32)

        ys = selective_scan_fn(
            xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, ssoflex,
            backend=selective_scan_backend).view(batch_size, k_group, -1, seq_len)

        if self.ig_mode == 'scan':
            y = self.ig_cross.merge(ys).view(batch_size, -1, height, width)
        else:
            y = cross_merge_fn(
                ys.view(batch_size, k_group, -1, height, width),
                in_channel_first=True,
                out_channel_first=True,
                scans=0,
                force_torch=False)
        if not channel_first:
            y = y.view(batch_size, -1, seq_len).transpose(1, 2).contiguous().view(
                batch_size, height, width, -1)
        y = out_norm(y)
        return y.to(x.dtype)

    def forwardv2(self, x: torch.Tensor, **kwargs):
        del kwargs
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

        importance = None
        if self.ig_mode == 'z_gate':
            importance = self.ig_scan_module.predict_importance(x)

        y = self.forward_core(x)
        y = self.out_act(y)
        if not self.disable_z:
            if self.ig_mode == 'z_gate' and importance is not None:
                z_importance = importance if self.channel_first else importance.permute(
                    0, 2, 3, 1).contiguous()
                z = apply_importance_to_z(
                    z,
                    z_importance,
                    self.gate_scale,
                    gate_mode=self.gate_mode)
            y = y * z
        return self.dropout(self.out_proj(y))
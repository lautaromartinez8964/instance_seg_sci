import torch
from mmdet.registry import MODELS

from mmdet.models.backbones.vmamba_official.vmamba import Backbone_VSSM


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
        **kwargs,
    ):
        # Disable parent preload and use explicit research loader for clarity.
        super().__init__(
            out_indices=out_indices,
            pretrained=None,
            norm_layer=norm_layer,
            **kwargs,
        )
        self.pretrained_key = pretrained_key
        self.strict_pretrained = strict_pretrained

        if official_pretrained:
            self.load_official_pretrained(
                official_pretrained,
                key=pretrained_key,
                strict=strict_pretrained,
            )

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

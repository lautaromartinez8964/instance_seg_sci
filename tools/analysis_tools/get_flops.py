# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import tempfile
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.config import Config, DictAction
from mmengine.logging import MMLogger
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.utils import digit_version

from mmdet.registry import MODELS

try:
    from mmengine.analysis import get_model_complexity_info
    from mmengine.analysis.print_helper import _format_size
except ImportError:
    raise ImportError('Please upgrade mmengine >= 0.6.0')


def parse_args():
    parser = argparse.ArgumentParser(description='Get a detector flops')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--num-images',
        type=int,
        default=100,
        help='num images of calculate model flops')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def inference(args, logger):
    if digit_version(torch.__version__) < digit_version('1.12'):
        logger.warning(
            'Some config files, such as configs/yolact and configs/detectors,'
            'may have compatibility issues with torch.jit when torch<1.12. '
            'If you want to calculate flops for these models, '
            'please make sure your pytorch version is >=1.12.')

    config_name = Path(args.config)
    if not config_name.exists():
        logger.error(f'{config_name} not found.')

    cfg = Config.fromfile(args.config)
    cfg.val_dataloader.batch_size = 1
    cfg.work_dir = tempfile.TemporaryDirectory().name

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # For some torch/triton/mamba_ssm version combinations, FLOPs tracing may
    # fail in causal_conv1d custom CUDA op due strict stride constraints.
    # We monkeypatch only in this analysis script with a numerically-compatible
    # torch implementation to keep architecture path and make FLOPs computable.
    try:
        import mamba_ssm.ops.triton.ssd_combined as mamba_ssd_combined

        def _safe_causal_conv1d_fwd(
                x,
                weight,
                bias,
                seq_idx=None,
                initial_states=None,
                final_states_out=None,
                silu_activation=False):
            # x: [B, D, L], weight: [D, W]
            w = weight.unsqueeze(1).to(dtype=x.dtype, device=x.device)
            b = bias.to(dtype=x.dtype, device=x.device) if bias is not None else None
            y = F.conv1d(x, w, bias=b, padding=w.shape[-1] - 1, groups=x.shape[1])
            y = y[..., :x.shape[-1]]
            if silu_activation:
                y = F.silu(y)
            return y

        mamba_ssd_combined.causal_conv1d_fwd_function = _safe_causal_conv1d_fwd

        # Also patch vendored VMamba path if present in this codebase.
        try:
            import mmdet.models.backbones.vmamba_official.mamba2.ssd_combined as vmamba_ssd_combined
            vmamba_ssd_combined.causal_conv1d_fwd_function = _safe_causal_conv1d_fwd
            vmamba_ssd_combined.causal_conv1d_fn = (
                lambda x, weight, bias, activation='silu':
                _safe_causal_conv1d_fwd(
                    x,
                    weight,
                    bias,
                    seq_idx=None,
                    initial_states=None,
                    final_states_out=None,
                    silu_activation=activation in ['silu', 'swish']))
        except Exception:
            pass

        # Patch causal_conv1d package symbols directly.
        try:
            import causal_conv1d
            import causal_conv1d.cpp_functions as causal_conv1d_cpp

            def _safe_causal_conv1d_fn(
                    x,
                    weight,
                    bias=None,
                    seq_idx=None,
                    initial_states=None,
                    return_final_states=False,
                    final_states_out=None,
                    activation='silu',
                    **kwargs):
                return _safe_causal_conv1d_fwd(
                    x,
                    weight,
                    bias,
                    seq_idx=seq_idx,
                    initial_states=initial_states,
                    final_states_out=final_states_out,
                    silu_activation=activation in ['silu', 'swish'])

            causal_conv1d.causal_conv1d_fn = _safe_causal_conv1d_fn
            causal_conv1d_cpp.causal_conv1d_fwd_function = _safe_causal_conv1d_fwd

            # mamba2 may bind causal_conv1d_fn at import time; patch there too.
            import mamba_ssm.modules.mamba2 as mamba2_module
            mamba2_module.causal_conv1d_fn = _safe_causal_conv1d_fn
        except Exception:
            pass
    except Exception as e:
        logger.warning(f'Failed to patch causal_conv1d fallback: {e}')

    init_default_scope(cfg.get('default_scope', 'mmdet'))

    # TODO: The following usage is temporary and not safe
    # use hard code to convert mmSyncBN to SyncBN. This is a known
    # bug in mmengine, mmSyncBN requires a distributed environment，
    # this question involves models like configs/strong_baselines
    if hasattr(cfg, 'head_norm_cfg'):
        cfg['head_norm_cfg'] = dict(type='SyncBN', requires_grad=True)
        cfg['model']['roi_head']['bbox_head']['norm_cfg'] = dict(
            type='SyncBN', requires_grad=True)
        cfg['model']['roi_head']['mask_head']['norm_cfg'] = dict(
            type='SyncBN', requires_grad=True)

    result = {}
    avg_flops = []
    data_loader = Runner.build_dataloader(cfg.val_dataloader)
    model = MODELS.build(cfg.model)
    if torch.cuda.is_available():
        model = model.cuda()
    model = revert_sync_batchnorm(model)
    model.eval()
    _forward = model.forward

    # Always compute params directly so we can still report it even if
    # JIT/Triton based FLOPs tracing fails for custom CUDA kernels.
    total_params = sum(p.numel() for p in model.parameters())

    for idx, data_batch in enumerate(data_loader):
        if idx == args.num_images:
            break
        data = model.data_preprocessor(data_batch)
        result['ori_shape'] = data['data_samples'][0].ori_shape
        result['pad_shape'] = data['data_samples'][0].pad_shape
        if hasattr(data['data_samples'][0], 'batch_input_shape'):
            result['pad_shape'] = data['data_samples'][0].batch_input_shape
        model.forward = partial(_forward, data_samples=data['data_samples'])
        try:
            outputs = get_model_complexity_info(
                model,
                None,
                inputs=data['inputs'],
                show_table=False,
                show_arch=False)
            avg_flops.append(outputs['flops'])
            result['compute_type'] = 'dataloader: load a picture from the dataset'
        except Exception as e:
            logger.warning(
                'FLOPs tracing failed, fallback to params-only mode. '
                f'Reason: {type(e).__name__}: {e}')
            result['compute_type'] = (
                'params-only fallback (FLOPs tracing failed on custom '
                'kernel/JIT path)')
            break
    del data_loader

    if len(avg_flops) > 0:
        mean_flops = _format_size(int(np.average(avg_flops)))
    else:
        mean_flops = 'N/A'
    params = _format_size(total_params)
    result['flops'] = mean_flops
    result['params'] = params

    return result


def main():
    args = parse_args()
    logger = MMLogger.get_instance(name='MMLogger')
    result = inference(args, logger)
    split_line = '=' * 30
    ori_shape = result['ori_shape']
    pad_shape = result['pad_shape']
    flops = result['flops']
    params = result['params']
    compute_type = result['compute_type']

    if pad_shape != ori_shape:
        print(f'{split_line}\nUse size divisor set input shape '
              f'from {ori_shape} to {pad_shape}')
    print(f'{split_line}\nCompute type: {compute_type}\n'
          f'Input shape: {pad_shape}\nFlops: {flops}\n'
          f'Params: {params}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify '
          'that the flops computation is correct.')


if __name__ == '__main__':
    main()

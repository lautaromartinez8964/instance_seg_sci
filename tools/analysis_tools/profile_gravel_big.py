"""
Profile gravel_big models: Params, FLOPs, FPS, latency, CUDA Memory.

Usage:
    conda activate mmdec
    python tools/analysis_tools/profile_gravel_big.py \
        configs/gravel_big/mask_rcnn_r50_fpn_36e_gravel_big.py \
        --input-size 640 640

For VMamba-based models, use the Triton-safe FLOPs script separately:
    python tools/analysis_tools/get_flops_vmamba_safe.py \
        configs/gravel_big/mask_rcnn_vmamba_official_2292_fpn_36e_gravel_big.py
"""

import argparse
import time
from functools import partial

import torch
from mmengine.config import Config
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope

from mmdet.registry import MODELS
from mmdet.apis import init_detector


def parse_args():
    parser = argparse.ArgumentParser(description='Profile a detector model')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--input-size', type=int, nargs=2, default=[640, 640],
                        help='input size (H W)')
    parser.add_argument('--warmup', type=int, default=20,
                        help='warmup iterations')
    parser.add_argument('--iter', type=int, default=100,
                        help='benchmark iterations')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmdet'))

    # Build model
    model = MODELS.build(cfg.model)
    model = revert_sync_batchnorm(model)
    model.cuda().eval()

    # 1. Params
    total_params = sum(p.numel() for p in model.parameters())
    params_m = total_params / 1e6
    print(f'[Params] {params_m:.3f} M ({total_params:,})')

    # 2. FLOPs — using mmengine's get_model_complexity_info with dummy input
    h, w = args.input_size
    try:
        from mmengine.analysis import get_model_complexity_info
        from mmengine.analysis.print_helper import _format_size

        dummy_input = torch.randn(1, 3, h, w).cuda()

        # Build a minimal data_sample for the model forward
        from mmdet.structures import DetDataSample
        data_sample = DetDataSample()
        data_sample.set_metainfo({
            'img_shape': (h, w),
            'ori_shape': (h, w),
            'pad_shape': (h, w),
            'batch_input_shape': (h, w),
            'scale_factor': (1.0, 1.0),
        })

        _forward = model.forward
        model.forward = partial(_forward, data_samples=[data_sample])

        outputs = get_model_complexity_info(
            model,
            None,
            inputs=dummy_input,
            show_table=False,
            show_arch=False)

        flops = outputs['flops']
        print(f'[FLOPs] {flops} (input {h}x{w})')

        model.forward = _forward
    except Exception as e:
        print(f'[FLOPs] Failed: {e}')
        print('[FLOPs] For VMamba models, use get_flops_vmamba_safe.py instead')

    # 3. FPS / Latency / CUDA Memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    dummy_img = torch.randn(1, 3, h, w).cuda()

    # Build data_sample for inference_detector style
    from mmdet.structures import DetDataSample
    data_sample = DetDataSample()
    data_sample.set_metainfo({
        'img_shape': (h, w),
        'ori_shape': (h, w),
        'pad_shape': (h, w),
        'batch_input_shape': (h, w),
        'scale_factor': (1.0, 1.0),
    })

    # Warmup
    with torch.no_grad():
        for _ in range(args.warmup):
            model.forward(dummy_img, data_samples=[data_sample])
    torch.cuda.synchronize()

    # Benchmark
    torch.cuda.reset_peak_memory_stats()
    times = []
    with torch.no_grad():
        for _ in range(args.iter):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.forward(dummy_img, data_samples=[data_sample])
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)

    peak_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB

    avg_time = sum(times) / len(times)
    fps = 1.0 / avg_time

    print(f'[FPS] {fps:.1f} img/s')
    print(f'[Latency] {avg_time * 1000:.1f} ms/img')
    print(f'[CUDA Memory] {peak_mem:.0f} MB (inference peak)')


if __name__ == '__main__':
    main()

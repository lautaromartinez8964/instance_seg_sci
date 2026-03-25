# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import tempfile
from functools import partial

from mmengine.analysis import get_model_complexity_info
from mmengine.analysis.print_helper import _format_size
from mmengine.config import Config
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner import Runner

import torch
import torch.nn as nn

from mmdet.registry import MODELS
from mmdet.models.backbones.vmamba_official.vmamba import Backbone_VSSM


def parse_args():
    parser = argparse.ArgumentParser(
        description='Get FLOPs for VMamba detector in a Triton-safe way')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--num-images', type=int, default=1)
    return parser.parse_args()


class DummyBackbone(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        b, _, h, w = x.shape
        dev = x.device
        dt = x.dtype
        return [
            torch.zeros((b, self.dims[0], h // 4, w // 4), device=dev, dtype=dt),
            torch.zeros((b, self.dims[1], h // 8, w // 8), device=dev, dtype=dt),
            torch.zeros((b, self.dims[2], h // 16, w // 16), device=dev, dtype=dt),
            torch.zeros((b, self.dims[3], h // 32, w // 32), device=dev, dtype=dt),
        ]


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.val_dataloader.batch_size = 1
    cfg.work_dir = tempfile.TemporaryDirectory().name

    init_default_scope(cfg.get('default_scope', 'mmdet'))

    loader = Runner.build_dataloader(cfg.val_dataloader)
    model = MODELS.build(cfg.model)
    model = revert_sync_batchnorm(model)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    # Extract backbone hyper-params from current config.
    bb_cfg = cfg.model.backbone
    dims = bb_cfg.get('dims', [48, 96, 192, 384])
    depths = bb_cfg.get('depths', [2, 2, 9, 2])

    # 1) Detector FLOPs without backbone (dummy backbone keeps neck+head path).
    model.backbone = DummyBackbone(dims)
    _forward = model.forward

    data_batch = next(iter(loader))
    data = model.data_preprocessor(data_batch)
    pad_shape = data['data_samples'][0].batch_input_shape

    model.forward = partial(_forward, data_samples=data['data_samples'])
    outputs = get_model_complexity_info(
        model,
        None,
        inputs=data['inputs'],
        show_table=False,
        show_arch=False)

    detector_wo_backbone_flops = int(outputs['flops'])
    detector_wo_backbone_params = int(outputs['params'])

    # 2) Backbone FLOPs (official VMamba analytical helper).
    backbone = Backbone_VSSM(dims=dims, depths=depths, out_indices=(0, 1, 2, 3))
    backbone_flops = int(backbone.flops(shape=(3, pad_shape[0], pad_shape[1]), verbose=False))
    backbone_params = sum(p.numel() for p in backbone.parameters())

    total_flops = detector_wo_backbone_flops + backbone_flops
    total_params = detector_wo_backbone_params + backbone_params

    print('=' * 30)
    print(f'Input shape: {pad_shape}')
    print(f'Backbone FLOPs: {_format_size(backbone_flops)}')
    print(f'Neck+Head FLOPs: {_format_size(detector_wo_backbone_flops)}')
    print(f'Total FLOPs: {_format_size(total_flops)}')
    print(f'Backbone Params: {_format_size(backbone_params)}')
    print(f'Neck+Head Params: {_format_size(detector_wo_backbone_params)}')
    print(f'Total Params: {_format_size(total_params)}')
    print('=' * 30)
    print('Note: This script is Triton-safe and intended for VMamba-style custom backbones.')


if __name__ == '__main__':
    main()

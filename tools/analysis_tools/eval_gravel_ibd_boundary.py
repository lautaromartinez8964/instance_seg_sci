import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.config import Config

from pycocotools.coco import COCO

from mmdet.apis import inference_detector, init_detector
from mmdet.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate IBD boundary precision/recall/F1 on gravel val split.')
    parser.add_argument('config', help='Model config file.')
    parser.add_argument('checkpoint', help='Checkpoint file.')
    parser.add_argument('--annotation', required=True, help='COCO annotation file.')
    parser.add_argument('--image-root', required=True, help='Validation image root.')
    parser.add_argument('--boundary-root', required=True, help='Boundary gt root directory.')
    parser.add_argument('--out-dir', help='Output directory. Defaults to <work_dir>/ibd_analysis/boundary_metrics_<checkpoint_name>.')
    parser.add_argument('--device', default='cuda:0', help='Inference device.')
    parser.add_argument('--threshold', type=float, default=0.5, help='Boundary probability threshold.')
    parser.add_argument('--top-k-hard', type=int, default=20, help='Number of hard cases to save.')
    return parser.parse_args()


def resolve_out_dir(config_path: str, checkpoint_path: str, out_dir: str | None) -> Path:
    if out_dir:
        return Path(out_dir)

    cfg = Config.fromfile(config_path)
    work_dir = cfg.get('work_dir')
    if not work_dir:
        raise ValueError(f'Config does not define work_dir: {config_path}')
    checkpoint_stem = Path(checkpoint_path).stem
    return Path(work_dir) / 'ibd_analysis' / f'boundary_metrics_{checkpoint_stem}'


def patch_torch_load():
    original_load = torch.load

    def unsafe_load(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return original_load(*args, **kwargs)

    torch.load = unsafe_load


def get_backbone(model):
    backbone = model.backbone
    if hasattr(backbone, 'module'):
        return backbone.module
    return backbone


def resolve_image_path(image_root: Path, file_name: str) -> Path:
    return image_root / Path(file_name).name


def load_boundary(boundary_root: Path, file_name: str) -> np.ndarray:
    boundary_path = boundary_root / f'{Path(file_name).stem}.png'
    boundary = mmcv.imread(str(boundary_path), flag='grayscale')
    if boundary is None:
        raise FileNotFoundError(f'Boundary map not found: {boundary_path.as_posix()}')
    return (boundary > 0).astype(np.uint8)


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def main():
    args = parse_args()
    patch_torch_load()
    register_all_modules()

    out_dir = resolve_out_dir(args.config, args.checkpoint, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    coco = COCO(args.annotation)
    image_root = Path(args.image_root)
    boundary_root = Path(args.boundary_root)

    ann_count = {}
    for ann in coco.dataset['annotations']:
        if ann.get('iscrowd', 0):
            continue
        ann_count[ann['image_id']] = ann_count.get(ann['image_id'], 0) + 1

    model = init_detector(args.config, args.checkpoint, device=args.device)
    backbone = get_backbone(model)
    if not hasattr(backbone, 'get_last_boundary_logits'):
        raise RuntimeError('Backbone does not expose get_last_boundary_logits().')

    total_tp = 0
    total_fp = 0
    total_fn = 0
    per_image = []

    for index, image_info in enumerate(coco.dataset['images'], start=1):
        image_path = resolve_image_path(image_root, image_info['file_name'])
        gt_boundary = load_boundary(boundary_root, image_info['file_name'])
        image_area = max(image_info['width'] * image_info['height'], 1)

        with torch.no_grad():
            inference_detector(model, str(image_path))
        logits = backbone.get_last_boundary_logits()
        if logits is None:
            raise RuntimeError(f'IBD logits not captured for image: {image_path.as_posix()}')

        probability = torch.sigmoid(logits.float())
        probability = F.interpolate(
            probability,
            size=gt_boundary.shape,
            mode='bilinear',
            align_corners=False).squeeze().detach().cpu().numpy()
        pred_boundary = (probability >= args.threshold).astype(np.uint8)

        tp = int((pred_boundary & gt_boundary).sum())
        fp = int((pred_boundary & (1 - gt_boundary)).sum())
        fn = int(((1 - pred_boundary) & gt_boundary).sum())
        precision = safe_ratio(tp, tp + fp)
        recall = safe_ratio(tp, tp + fn)
        f1 = safe_ratio(2 * precision * recall, precision + recall)
        dice = safe_ratio(2 * tp, 2 * tp + fp + fn)
        iou = safe_ratio(tp, tp + fp + fn)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        per_image.append(
            dict(
                index=index,
                image_id=image_info['id'],
                file_name=image_info['file_name'],
                width=image_info['width'],
                height=image_info['height'],
                instance_count=ann_count.get(image_info['id'], 0),
                density_per_10k=ann_count.get(image_info['id'], 0) / image_area * 10000.0,
                precision=precision,
                recall=recall,
                f1=f1,
                dice=dice,
                iou=iou,
                gt_positive=int(gt_boundary.sum()),
                pred_positive=int(pred_boundary.sum()),
                pred_mean=float(probability.mean()),
                pred_max=float(probability.max()),
            ))

        if index % 25 == 0:
            print(f'processed={index}/{len(coco.dataset["images"])}')

    micro_precision = safe_ratio(total_tp, total_tp + total_fp)
    micro_recall = safe_ratio(total_tp, total_tp + total_fn)
    micro_f1 = safe_ratio(2 * micro_precision * micro_recall, micro_precision + micro_recall)
    micro_dice = safe_ratio(2 * total_tp, 2 * total_tp + total_fp + total_fn)
    micro_iou = safe_ratio(total_tp, total_tp + total_fp + total_fn)

    macro_precision = float(np.mean([item['precision'] for item in per_image]))
    macro_recall = float(np.mean([item['recall'] for item in per_image]))
    macro_f1 = float(np.mean([item['f1'] for item in per_image]))
    macro_dice = float(np.mean([item['dice'] for item in per_image]))
    macro_iou = float(np.mean([item['iou'] for item in per_image]))

    hard_cases = sorted(per_image, key=lambda item: (item['f1'], item['recall'], item['precision']))[:args.top_k_hard]
    summary = dict(
        config=args.config,
        checkpoint=args.checkpoint,
        threshold=args.threshold,
        num_images=len(per_image),
        micro=dict(
            precision=micro_precision,
            recall=micro_recall,
            f1=micro_f1,
            dice=micro_dice,
            iou=micro_iou,
            tp=total_tp,
            fp=total_fp,
            fn=total_fn,
        ),
        macro=dict(
            precision=macro_precision,
            recall=macro_recall,
            f1=macro_f1,
            dice=macro_dice,
            iou=macro_iou,
        ),
        hard_cases=hard_cases,
    )

    with (out_dir / 'boundary_metrics_summary.json').open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    with (out_dir / 'boundary_metrics_per_image.json').open('w', encoding='utf-8') as handle:
        json.dump(per_image, handle, indent=2, ensure_ascii=False)

    print(f'micro_precision={micro_precision:.6f}')
    print(f'micro_recall={micro_recall:.6f}')
    print(f'micro_f1={micro_f1:.6f}')
    print(f'out_dir={out_dir.as_posix()}')


if __name__ == '__main__':
    main()
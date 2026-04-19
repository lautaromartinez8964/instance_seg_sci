import argparse
import contextlib
import io
import json
import os
from pathlib import Path

os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')

import numpy as np
import torch
from mmengine.config import Config
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from mmdet.apis import inference_detector, init_detector
from mmdet.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(description='Compare segmentation mAP by density buckets for two gravel models.')
    parser.add_argument('config_a', help='Model A config.')
    parser.add_argument('checkpoint_a', help='Model A checkpoint.')
    parser.add_argument('config_b', help='Model B config.')
    parser.add_argument('checkpoint_b', help='Model B checkpoint.')
    parser.add_argument('--annotation', required=True, help='COCO annotation file.')
    parser.add_argument('--image-root', required=True, help='Validation image root.')
    parser.add_argument('--out-dir', help='Output directory. Defaults to <config_a.work_dir>/ibd_analysis/density_bucket_compare_<ckpt_a>_vs_<ckpt_b>.')
    parser.add_argument('--device', default='cuda:0', help='Inference device.')
    parser.add_argument('--score-thr', type=float, default=0.001, help='Prediction score threshold.')
    parser.add_argument('--reuse-cache', action='store_true', help='Reuse cached prediction json if available.')
    return parser.parse_args()


def resolve_out_dir(config_path: str, checkpoint_a: str, checkpoint_b: str,
                    out_dir: str | None) -> Path:
    if out_dir:
        return Path(out_dir)

    cfg = Config.fromfile(config_path)
    work_dir = cfg.get('work_dir')
    if not work_dir:
        raise ValueError(f'Config does not define work_dir: {config_path}')
    stem_a = Path(checkpoint_a).stem
    stem_b = Path(checkpoint_b).stem
    return Path(work_dir) / 'ibd_analysis' / f'density_bucket_compare_{stem_a}_vs_{stem_b}'


def patch_torch_load():
    original_load = torch.load

    def unsafe_load(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return original_load(*args, **kwargs)

    torch.load = unsafe_load


def sanitize_label(text: str) -> str:
    return ''.join(ch if ch.isalnum() else '_' for ch in text).strip('_').lower()


def category_id_lookup(coco_gt: COCO):
    categories = sorted(coco_gt.dataset['categories'], key=lambda item: item['id'])
    return [category['id'] for category in categories]


def encode_mask(mask: np.ndarray):
    encoded = mask_utils.encode(np.asfortranarray(mask[:, :, None].astype(np.uint8)))[0]
    encoded['counts'] = encoded['counts'].decode('utf-8')
    return encoded


def run_inference_cache(config_path: str, checkpoint_path: str, label: str,
                        coco_gt: COCO, image_root: Path, out_dir: Path,
                        device: str, score_thr: float, reuse_cache: bool):
    cache_path = out_dir / f'{sanitize_label(label)}_predictions.json'
    if reuse_cache and cache_path.exists():
        with cache_path.open('r', encoding='utf-8') as handle:
            return json.load(handle), cache_path

    model = init_detector(config_path, checkpoint_path, device=device)
    cat_ids = category_id_lookup(coco_gt)
    results = []

    for index, image_info in enumerate(coco_gt.dataset['images'], start=1):
        image_path = image_root / Path(image_info['file_name']).name
        with torch.no_grad():
            result = inference_detector(model, str(image_path))
        pred = result.pred_instances
        if len(pred) == 0:
            continue

        scores = pred.scores.detach().cpu().numpy()
        labels = pred.labels.detach().cpu().numpy()
        bboxes = pred.bboxes.detach().cpu().numpy()
        masks = pred.masks.detach().cpu().numpy().astype(np.uint8)

        keep = scores >= score_thr
        scores = scores[keep]
        labels = labels[keep]
        bboxes = bboxes[keep]
        masks = masks[keep]
        for score_value, label_value, bbox_value, mask in zip(
                scores.tolist(), labels.tolist(), bboxes.tolist(), masks):
            if int(label_value) >= len(cat_ids):
                continue
            x1, y1, x2, y2 = bbox_value
            results.append(
                dict(
                    image_id=image_info['id'],
                    category_id=cat_ids[int(label_value)],
                    segmentation=encode_mask(mask),
                    score=float(score_value),
                    bbox=[float(x1), float(y1), float(max(x2 - x1, 0.0)), float(max(y2 - y1, 0.0))],
                ))

        if index % 25 == 0:
            print(f'{label}: processed {index}/{len(coco_gt.dataset["images"])}')

    with cache_path.open('w', encoding='utf-8') as handle:
        json.dump(results, handle)
    return results, cache_path


def build_density_buckets(coco_gt: COCO):
    ann_count = {}
    for ann in coco_gt.dataset['annotations']:
        if ann.get('iscrowd', 0):
            continue
        ann_count[ann['image_id']] = ann_count.get(ann['image_id'], 0) + 1

    image_records = []
    for image_info in coco_gt.dataset['images']:
        area = max(int(image_info['width']) * int(image_info['height']), 1)
        count = ann_count.get(image_info['id'], 0)
        density = count / area * 10000.0
        image_records.append(
            dict(
                image_id=image_info['id'],
                file_name=image_info['file_name'],
                density_per_10k=density,
                instance_count=count,
            ))
    image_records.sort(key=lambda item: item['density_per_10k'])

    num_images = len(image_records)
    low_end = num_images // 3
    mid_end = (num_images * 2) // 3
    buckets = {
        'low': image_records[:low_end],
        'mid': image_records[low_end:mid_end],
        'high': image_records[mid_end:],
        'all': image_records,
    }
    return buckets


def evaluate_bucket(coco_gt: COCO, predictions, image_ids):
    if predictions:
        coco_dt = coco_gt.loadRes(predictions)
    else:
        coco_dt = coco_gt.loadRes([])
    evaluator = COCOeval(coco_gt, coco_dt, 'segm')
    evaluator.params.imgIds = list(image_ids)
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    stats = evaluator.stats.tolist()
    return dict(
        segm_mAP=stats[0],
        segm_mAP_50=stats[1],
        segm_mAP_75=stats[2],
        segm_mAP_s=stats[3],
        segm_mAP_m=stats[4],
        segm_mAP_l=stats[5],
    )


def main():
    args = parse_args()
    patch_torch_load()
    register_all_modules()

    out_dir = resolve_out_dir(
        args.config_a,
        args.checkpoint_a,
        args.checkpoint_b,
        args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_root = Path(args.image_root)

    coco_gt = COCO(args.annotation)
    buckets = build_density_buckets(coco_gt)

    predictions_a, cache_a = run_inference_cache(
        args.config_a, args.checkpoint_a, 'model_a', coco_gt, image_root,
        out_dir, args.device, args.score_thr, args.reuse_cache)
    predictions_b, cache_b = run_inference_cache(
        args.config_b, args.checkpoint_b, 'model_b', coco_gt, image_root,
        out_dir, args.device, args.score_thr, args.reuse_cache)

    summary = dict(
        model_a=dict(config=args.config_a, checkpoint=args.checkpoint_a, cache=str(cache_a)),
        model_b=dict(config=args.config_b, checkpoint=args.checkpoint_b, cache=str(cache_b)),
        buckets={},
    )

    for bucket_name, records in buckets.items():
        image_ids = [item['image_id'] for item in records]
        densities = [item['density_per_10k'] for item in records]
        stats_a = evaluate_bucket(coco_gt, predictions_a, image_ids)
        stats_b = evaluate_bucket(coco_gt, predictions_b, image_ids)
        summary['buckets'][bucket_name] = dict(
            num_images=len(records),
            density_min=min(densities) if densities else None,
            density_max=max(densities) if densities else None,
            density_mean=float(np.mean(densities)) if densities else None,
            model_a=stats_a,
            model_b=stats_b,
            delta=dict(
                segm_mAP=stats_a['segm_mAP'] - stats_b['segm_mAP'],
                segm_mAP_50=stats_a['segm_mAP_50'] - stats_b['segm_mAP_50'],
                segm_mAP_75=stats_a['segm_mAP_75'] - stats_b['segm_mAP_75'],
            ),
        )

    with (out_dir / 'density_bucket_compare_summary.json').open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f'out_dir={out_dir.as_posix()}')
    for bucket_name, bucket in summary['buckets'].items():
        print(
            f'{bucket_name}: ibd={bucket["model_a"]["segm_mAP"]:.4f} '
            f'baseline={bucket["model_b"]["segm_mAP"]:.4f} '
            f'delta={bucket["delta"]["segm_mAP"]:+.4f}')


if __name__ == '__main__':
    main()
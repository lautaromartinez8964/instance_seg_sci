"""Quantitative detection error analysis for gravel instance segmentation.

Produces per-image and aggregate statistics:
- False Negatives (missed GT instances)
- False Positives (spurious predictions)
- Fragmentation (one GT matched by multiple preds)
- Merging (one pred matched by multiple GTs)
- IoU quality of matched pairs
- Size-stratified analysis (small / medium / large GT instances)
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')

import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils

from mmengine.config import Config
from mmdet.apis import inference_detector, init_detector
from mmdet.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(description='Gravel detection error analysis')
    parser.add_argument('config', help='Model config file')
    parser.add_argument('checkpoint', help='Model checkpoint file')
    parser.add_argument('--annotation', required=True, help='COCO annotation file')
    parser.add_argument('--image-root', required=True, help='Image root directory')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--score-thr', type=float, default=0.3,
                        help='Prediction score threshold')
    parser.add_argument('--iou-thr', type=float, default=0.5,
                        help='IoU threshold for matching pred to GT')
    parser.add_argument('--max-images', type=int, default=0,
                        help='Max images to analyze (0=all)')
    parser.add_argument('--out-dir', required=True)
    return parser.parse_args()


def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def decode_rle_mask(rle, h: int, w: int) -> np.ndarray:
    """Decode RLE to binary mask."""
    return mask_utils.decode(rle).astype(bool)


def analyze_image(gt_masks, gt_areas, pred_masks, pred_scores, iou_thr=0.5):
    """Analyze detection errors for a single image.

    Returns dict with:
        num_gt, num_pred,
        num_fn (missed GT), num_fp (spurious pred),
        num_fragmented_gt (1 GT matched by >1 pred),
        num_merged_pred (1 pred matched by >1 GT),
        matched_ious (list of IoU for matched pairs),
        unmatched_gt_areas (areas of missed GTs),
        unmatched_pred_scores (scores of FPs),
        gt_size_bins (small/medium/large counts for GT),
        fn_size_bins (missed GT by size),
    """
    num_gt = len(gt_masks)
    num_pred = len(pred_masks)

    if num_gt == 0 and num_pred == 0:
        return dict(num_gt=0, num_pred=0, num_fn=0, num_fp=0,
                    num_fragmented_gt=0, num_merged_pred=0,
                    matched_ious=[], unmatched_gt_areas=[],
                    unmatched_pred_scores=[], gt_size_bins={}, fn_size_bins={})

    # Build IoU matrix: [num_gt, num_pred]
    iou_matrix = np.zeros((num_gt, num_pred), dtype=np.float32)
    for i in range(num_gt):
        for j in range(num_pred):
            iou_matrix[i, j] = compute_mask_iou(gt_masks[i], pred_masks[j])

    # Greedy matching: match highest IoU pairs first
    matched_gt = set()
    matched_pred = set()
    matched_ious = []

    # Get all IoU pairs sorted descending
    pairs = []
    for i in range(num_gt):
        for j in range(num_pred):
            if iou_matrix[i, j] >= iou_thr:
                pairs.append((iou_matrix[i, j], i, j))
    pairs.sort(reverse=True)

    for iou_val, gi, pj in pairs:
        if gi in matched_gt or pj in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pj)
        matched_ious.append(iou_val)

    # Count fragmentation: how many GTs are matched by >1 pred
    gt_to_preds = defaultdict(list)
    pred_to_gts = defaultdict(list)
    for iou_val, gi, pj in pairs:
        gt_to_preds[gi].append((iou_val, pj))
        pred_to_gts[pj].append((iou_val, gi))

    num_fragmented_gt = sum(1 for gi, plist in gt_to_preds.items()
                            if len(plist) > 1 and gi in matched_gt)
    num_merged_pred = sum(1 for pj, glist in pred_to_gts.items()
                          if len(glist) > 1 and pj in matched_pred)

    # False negatives: GTs not matched
    fn_indices = set(range(num_gt)) - matched_gt
    num_fn = len(fn_indices)
    unmatched_gt_areas = [float(gt_areas[i]) for i in fn_indices]

    # False positives: preds not matched
    fp_indices = set(range(num_pred)) - matched_pred
    num_fp = len(fp_indices)
    unmatched_pred_scores = [float(pred_scores[j]) for j in fp_indices]

    # Size-stratified analysis (COCO-like: small<32², medium<96², large>=96²)
    # For 640x640 images, use pixel area thresholds
    SMALL_THR = 32 * 32   # 1024
    MEDIUM_THR = 96 * 96  # 9216

    gt_size_bins = {'small': 0, 'medium': 0, 'large': 0}
    fn_size_bins = {'small': 0, 'medium': 0, 'large': 0}

    for i in range(num_gt):
        area = gt_areas[i]
        if area < SMALL_THR:
            bin_name = 'small'
        elif area < MEDIUM_THR:
            bin_name = 'medium'
        else:
            bin_name = 'large'
        gt_size_bins[bin_name] += 1
        if i in fn_indices:
            fn_size_bins[bin_name] += 1

    return dict(
        num_gt=num_gt, num_pred=num_pred,
        num_fn=num_fn, num_fp=num_fp,
        num_fragmented_gt=num_fragmented_gt,
        num_merged_pred=num_merged_pred,
        matched_ious=matched_ious,
        unmatched_gt_areas=unmatched_gt_areas,
        unmatched_pred_scores=unmatched_pred_scores,
        gt_size_bins=gt_size_bins,
        fn_size_bins=fn_size_bins,
    )


def main():
    args = parse_args()
    register_all_modules()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    coco = COCO(args.annotation)
    images = coco.dataset['images']
    if args.max_images > 0:
        images = images[:args.max_images]

    model = init_detector(args.config, args.checkpoint, device=args.device)

    # Aggregate stats
    agg = dict(
        total_images=0,
        total_gt=0, total_pred=0,
        total_fn=0, total_fp=0,
        total_fragmented_gt=0, total_merged_pred=0,
        all_matched_ious=[],
        all_unmatched_gt_areas=[],
        all_unmatched_pred_scores=[],
        gt_size_bins={'small': 0, 'medium': 0, 'large': 0},
        fn_size_bins={'small': 0, 'medium': 0, 'large': 0},
    )
    per_image = []

    total = len(images)
    for index, image_info in enumerate(images, start=1):
        image_path = Path(args.image_root) / Path(image_info['file_name']).name
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[image_info['id']]))

        # GT masks and areas
        gt_masks = []
        gt_areas = []
        for ann in anns:
            if 'segmentation' not in ann or ann.get('iscrowd', 0):
                continue
            rle = coco.annToRLE(ann)
            h, w = image_info['height'], image_info['width']
            mask = decode_rle_mask(rle, h, w)
            gt_masks.append(mask)
            gt_areas.append(ann.get('area', float(mask.sum())))

        # Predictions
        result = inference_detector(model, str(image_path))
        pred_instance = result.pred_instances

        pred_masks_list = []
        pred_scores_list = []
        if hasattr(pred_instance, 'masks') and pred_instance.masks is not None:
            scores = pred_instance.scores.cpu().numpy()
            masks = pred_instance.masks.cpu().numpy()  # [N, H, W]

            # Filter by score threshold
            keep = scores >= args.score_thr
            scores = scores[keep]
            masks = masks[keep]

            for j in range(len(scores)):
                pred_masks_list.append(masks[j].astype(bool))
                pred_scores_list.append(float(scores[j]))

        # Analyze
        stats = analyze_image(
            gt_masks, gt_areas, pred_masks_list, pred_scores_list,
            iou_thr=args.iou_thr)

        # Aggregate
        agg['total_images'] += 1
        agg['total_gt'] += stats['num_gt']
        agg['total_pred'] += stats['num_pred']
        agg['total_fn'] += stats['num_fn']
        agg['total_fp'] += stats['num_fp']
        agg['total_fragmented_gt'] += stats['num_fragmented_gt']
        agg['total_merged_pred'] += stats['num_merged_pred']
        agg['all_matched_ious'].extend(stats['matched_ious'])
        agg['all_unmatched_gt_areas'].extend(stats['unmatched_gt_areas'])
        agg['all_unmatched_pred_scores'].extend(stats['unmatched_pred_scores'])
        for k in ['small', 'medium', 'large']:
            agg['gt_size_bins'][k] += stats['gt_size_bins'].get(k, 0)
            agg['fn_size_bins'][k] += stats['fn_size_bins'].get(k, 0)

        per_image.append({
            'image': image_info['file_name'],
            'img_id': image_info['id'],
            **stats,
            'matched_ious': stats['matched_ious'],
            'unmatched_gt_areas': stats['unmatched_gt_areas'],
            'unmatched_pred_scores': stats['unmatched_pred_scores'],
        })

        if index % 10 == 0 or index == total:
            print(f'[{index}/{total}] '
                  f'GT={stats["num_gt"]} Pred={stats["num_pred"]} '
                  f'FN={stats["num_fn"]} FP={stats["num_fp"]} '
                  f'Frag={stats["num_fragmented_gt"]} Merge={stats["num_merged_pred"]}')

    # Compute summary metrics
    matched_ious = np.array(agg['all_matched_ious'])
    unmatched_gt_areas = np.array(agg['all_unmatched_gt_areas'])
    unmatched_pred_scores = np.array(agg['all_unmatched_pred_scores'])

    recall = (agg['total_gt'] - agg['total_fn']) / max(agg['total_gt'], 1)
    precision = (agg['total_pred'] - agg['total_fp']) / max(agg['total_pred'], 1)

    summary = {
        'total_images': agg['total_images'],
        'total_gt_instances': agg['total_gt'],
        'total_pred_instances': agg['total_pred'],
        'total_fn (missed GT)': agg['total_fn'],
        'total_fp (spurious pred)': agg['total_fp'],
        'recall (IoU>={:.1f})'.format(args.iou_thr): round(recall, 4),
        'precision (IoU>={:.1f})'.format(args.iou_thr): round(precision, 4),
        'fragmented_gt (1 GT -> multi pred)': agg['total_fragmented_gt'],
        'merged_pred (1 pred <- multi GT)': agg['total_merged_pred'],
        'mean_matched_iou': round(float(matched_ious.mean()), 4) if len(matched_ious) else 0,
        'median_matched_iou': round(float(np.median(matched_ious)), 4) if len(matched_ious) else 0,
        'gt_size_distribution': agg['gt_size_bins'],
        'fn_by_size': agg['fn_size_bins'],
        'fn_rate_by_size': {},
        'mean_fn_area': round(float(unmatched_gt_areas.mean()), 1) if len(unmatched_gt_areas) else 0,
        'mean_fp_score': round(float(unmatched_pred_scores.mean()), 4) if len(unmatched_pred_scores) else 0,
    }

    # FN rate by size
    for k in ['small', 'medium', 'large']:
        gt_count = agg['gt_size_bins'][k]
        fn_count = agg['fn_size_bins'][k]
        summary['fn_rate_by_size'][k] = round(fn_count / max(gt_count, 1), 4)

    # Print summary
    print('\n' + '=' * 60)
    print('DETECTION ERROR ANALYSIS SUMMARY')
    print('=' * 60)
    for k, v in summary.items():
        if isinstance(v, dict):
            print(f'  {k}:')
            for kk, vv in v.items():
                print(f'    {kk}: {vv}')
        else:
            print(f'  {k}: {v}')
    print('=' * 60)

    # Save
    with open(out_dir / 'error_analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(out_dir / 'error_analysis_per_image.json', 'w') as f:
        # Convert numpy types for JSON
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        json.dump(per_image, f, indent=2, ensure_ascii=False, default=convert)

    print(f'\nResults saved to {out_dir}')


if __name__ == '__main__':
    main()

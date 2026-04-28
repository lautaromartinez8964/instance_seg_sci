#!/usr/bin/env python3
"""Sweep NWD RPN assigner thresholds on a trained checkpoint.

This script keeps the detector checkpoint fixed and sweeps the RPN assigner
hyperparameters used during positive/negative sample assignment.

It reports three groups of evidence for each setting:
1) RPN positive sample counts
2) RPN proposal recall on small objects
3) Proposal noise ratios

The recall/noise metrics are invariant to assigner thresholds for a fixed
checkpoint, but they are included so the calibration table stays complete.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
from pathlib import Path
from statistics import mean, pstdev

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.structures import InstanceData
from mmengine.utils import import_modules_from_strings

from mmdet.apis import init_detector
from mmdet.models.task_modules.prior_generators import anchor_inside_flags
from mmdet.registry import DATASETS, TASK_UTILS
from mmdet.structures.bbox import bbox_overlaps, cat_boxes, get_box_tensor
from mmdet.utils import register_all_modules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Sweep RPN NWD assigner thresholds on a fixed checkpoint')
    parser.add_argument('config', help='Config path')
    parser.add_argument('checkpoint', help='Checkpoint path')
    parser.add_argument('--device', default='cuda:0', help='Device')
    parser.add_argument('--max-images', type=int, default=0,
                        help='Max images to analyze; 0 means all val images')
    parser.add_argument('--constant', type=float, default=8.0,
                        help='NWD normalization constant')
    parser.add_argument('--pos-thrs', type=float, nargs='+', default=[0.4, 0.5, 0.6],
                        help='Positive thresholds to sweep')
    parser.add_argument('--neg-thrs', type=float, nargs='+', default=[0.1, 0.2, 0.3],
                        help='Negative thresholds to sweep')
    parser.add_argument('--out', required=True, help='Output markdown path')
    parser.add_argument('--json-out', default=None,
                        help='Optional JSON output path')
    return parser.parse_args()


def safe_mean(values: list[float]) -> float:
    values = [v for v in values if not math.isnan(v)]
    return float(mean(values)) if values else float('nan')


def safe_std(values: list[float]) -> float:
    values = [v for v in values if not math.isnan(v)]
    return float(pstdev(values)) if len(values) > 1 else 0.0


def compute_ar(topk_props: torch.Tensor,
               gt_bboxes: torch.Tensor,
               iou_thresholds: list[float]) -> float:
    if gt_bboxes.numel() == 0:
        return float('nan')
    if topk_props.numel() == 0:
        return 0.0

    ious = bbox_overlaps(topk_props, gt_bboxes)
    if ious.numel() == 0:
        return 0.0

    gt_max = ious.max(dim=0).values
    recalls = [float((gt_max >= thr).float().mean().item())
               for thr in iou_thresholds]
    return float(sum(recalls) / len(recalls))


def build_assigner(constant: float, pos_thr: float,
                   neg_thr: float) -> object:
    return TASK_UTILS.build(dict(
        type='MaxIoUAssigner',
        iou_calculator=dict(type='BboxOverlaps2D_NWD', constant=constant),
        pos_iou_thr=pos_thr,
        neg_iou_thr=neg_thr,
        min_pos_iou=neg_thr,
        match_low_quality=True,
        ignore_iof_thr=-1))


def collect_assigner_stats(model,
                           assigner,
                           cls_scores: list[torch.Tensor],
                           batch_data_sample) -> dict[str, float]:
    img_meta = batch_data_sample.metainfo
    gt_instances = batch_data_sample.gt_instances

    featmap_sizes = [feat.shape[-2:] for feat in cls_scores]
    anchor_list, valid_flag_list = model.rpn_head.get_anchors(
        featmap_sizes=featmap_sizes,
        batch_img_metas=[img_meta],
        device=cls_scores[0].device)

    flat_anchors = cat_boxes(anchor_list[0])
    valid_flags = torch.cat(valid_flag_list[0])
    inside_flags = anchor_inside_flags(
        flat_anchors,
        valid_flags,
        img_meta['img_shape'][:2],
        model.rpn_head.train_cfg['allowed_border'])

    if not inside_flags.any():
        return {
            'num_pos_assigned': 0.0,
            'num_pos_sampled': 0.0,
            'num_small_pos_assigned': 0.0,
            'num_small_pos_sampled': 0.0,
            'pos_overlap_assigned_mean': float('nan'),
            'pos_overlap_sampled_mean': float('nan'),
        }

    priors = flat_anchors[inside_flags]
    pred_instances = InstanceData(priors=priors)
    assign_result = assigner.assign(
        pred_instances=pred_instances,
        gt_instances=gt_instances,
        gt_instances_ignore=None)

    assigned_pos_inds = torch.nonzero(assign_result.gt_inds > 0,
                                      as_tuple=False).squeeze(1)
    sampling_result = model.rpn_head.sampler.sample(
        assign_result=assign_result,
        pred_instances=pred_instances,
        gt_instances=gt_instances)

    sampled_pos_inds = sampling_result.pos_inds

    gt_bboxes = get_box_tensor(gt_instances.bboxes)
    gt_areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (gt_bboxes[:, 3] - gt_bboxes[:, 1])
    small_thr = 32.0 * 32.0

    if assigned_pos_inds.numel() > 0:
        assigned_gt_inds = assign_result.gt_inds[assigned_pos_inds] - 1
        assigned_small = (gt_areas[assigned_gt_inds] < small_thr).sum().item()
        assigned_overlaps = assign_result.max_overlaps[assigned_pos_inds]
        assigned_overlap_mean = float(assigned_overlaps.mean().item())
    else:
        assigned_small = 0.0
        assigned_overlap_mean = float('nan')

    if sampled_pos_inds.numel() > 0:
        sampled_gt_inds = sampling_result.pos_assigned_gt_inds
        sampled_small = (gt_areas[sampled_gt_inds] < small_thr).sum().item()
        sampled_overlaps = assign_result.max_overlaps[sampled_pos_inds]
        sampled_overlap_mean = float(sampled_overlaps.mean().item())
    else:
        sampled_small = 0.0
        sampled_overlap_mean = float('nan')

    return {
        'num_pos_assigned': float(assigned_pos_inds.numel()),
        'num_pos_sampled': float(sampled_pos_inds.numel()),
        'num_small_pos_assigned': float(assigned_small),
        'num_small_pos_sampled': float(sampled_small),
        'pos_overlap_assigned_mean': assigned_overlap_mean,
        'pos_overlap_sampled_mean': sampled_overlap_mean,
    }


def main() -> None:
    args = parse_args()
    register_all_modules(init_default_scope=False)

    cfg = Config.fromfile(args.config)
    if cfg.get('custom_imports', None):
        import_modules_from_strings(**cfg.custom_imports)
    init_default_scope(cfg.get('default_scope', 'mmdet'))

    model = init_detector(cfg, args.checkpoint, device=args.device)
    model.eval()

    dataset_cfg = copy.deepcopy(cfg.val_dataloader.dataset)
    dataset = DATASETS.build(dataset_cfg)
    total_images = len(dataset) if args.max_images <= 0 else min(args.max_images, len(dataset))

    sweep_settings = [
        (args.constant, pos_thr, neg_thr)
        for pos_thr, neg_thr in itertools.product(args.pos_thrs, args.neg_thrs)
        if neg_thr < pos_thr
    ]

    if not sweep_settings:
        raise ValueError('No valid sweep settings: ensure neg_thrs < pos_thrs.')

    iou_thresholds = [x / 100 for x in range(50, 100, 5)]
    small_thr = 32.0 * 32.0

    proposal_ar100_all = []
    proposal_ar1000_all = []
    proposal_ar100_small = []
    proposal_ar1000_small = []
    proposal_noise_ratio_03 = []
    proposal_noise_ratio_05 = []
    proposal_match_ratio_05 = []

    stats_by_setting = {
        setting: {
            'num_pos_assigned': [],
            'num_pos_sampled': [],
            'num_small_pos_assigned': [],
            'num_small_pos_sampled': [],
            'pos_overlap_assigned_mean': [],
            'pos_overlap_sampled_mean': [],
        }
        for setting in sweep_settings
    }

    with torch.no_grad():
        for idx in range(total_images):
            item = dataset[idx]
            data_batch = {
                'inputs': [item['inputs']],
                'data_samples': [item['data_samples']]
            }
            packed = model.data_preprocessor(data_batch, training=False)
            batch_inputs = packed['inputs']
            batch_data_sample = packed['data_samples'][0]

            feats = model.extract_feat(batch_inputs)
            cls_scores, bbox_preds = model.rpn_head(feats)

            proposal_list = model.rpn_head.predict_by_feat(
                cls_scores=cls_scores,
                bbox_preds=bbox_preds,
                batch_img_metas=[batch_data_sample.metainfo],
                cfg=model.test_cfg.rpn,
                rescale=False)
            proposals = proposal_list[0]
            proposal_bboxes = get_box_tensor(proposals.bboxes)

            gt_bboxes = get_box_tensor(batch_data_sample.gt_instances.bboxes)
            gt_areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (gt_bboxes[:, 3] - gt_bboxes[:, 1]) \
                if gt_bboxes.numel() > 0 else torch.empty(0, device=proposal_bboxes.device)
            gt_bboxes_small = gt_bboxes[gt_areas < small_thr] if gt_areas.numel() > 0 else gt_bboxes

            top100 = proposal_bboxes[:100]
            top1000 = proposal_bboxes[:1000]

            proposal_ar100_all.append(compute_ar(top100, gt_bboxes, iou_thresholds))
            proposal_ar1000_all.append(compute_ar(top1000, gt_bboxes, iou_thresholds))
            proposal_ar100_small.append(compute_ar(top100, gt_bboxes_small, iou_thresholds))
            proposal_ar1000_small.append(compute_ar(top1000, gt_bboxes_small, iou_thresholds))

            if gt_bboxes.numel() == 0 or top1000.numel() == 0:
                proposal_noise_ratio_03.append(float('nan'))
                proposal_noise_ratio_05.append(float('nan'))
                proposal_match_ratio_05.append(float('nan'))
            else:
                ious = bbox_overlaps(top1000, gt_bboxes)
                max_iou_per_prop = ious.max(dim=1).values
                proposal_noise_ratio_03.append(float((max_iou_per_prop < 0.3).float().mean().item()))
                proposal_noise_ratio_05.append(float((max_iou_per_prop < 0.5).float().mean().item()))
                proposal_match_ratio_05.append(float((max_iou_per_prop >= 0.5).float().mean().item()))

            for setting in sweep_settings:
                constant, pos_thr, neg_thr = setting
                assigner = build_assigner(constant, pos_thr, neg_thr)
                assign_stats = collect_assigner_stats(model, assigner, cls_scores, batch_data_sample)
                for key, value in assign_stats.items():
                    stats_by_setting[setting][key].append(value)

            if (idx + 1) % 50 == 0 or (idx + 1) == total_images:
                print(f'[{idx + 1}/{total_images}] processed')

    results = []
    for constant, pos_thr, neg_thr in sweep_settings:
        item = {
            'constant': constant,
            'pos_iou_thr': pos_thr,
            'neg_iou_thr': neg_thr,
            'min_pos_iou': neg_thr,
            'num_images': total_images,
            'evidence_1_rpn_recall': {
                'AR@100_all': safe_mean(proposal_ar100_all),
                'AR@1000_all': safe_mean(proposal_ar1000_all),
                'AR@100_small': safe_mean(proposal_ar100_small),
                'AR@1000_small': safe_mean(proposal_ar1000_small),
            },
            'evidence_2_rpn_pos_distribution': {
                'num_pos_assigned_mean': safe_mean(stats_by_setting[(constant, pos_thr, neg_thr)]['num_pos_assigned']),
                'num_pos_sampled_mean': safe_mean(stats_by_setting[(constant, pos_thr, neg_thr)]['num_pos_sampled']),
                'num_small_pos_assigned_mean': safe_mean(stats_by_setting[(constant, pos_thr, neg_thr)]['num_small_pos_assigned']),
                'num_small_pos_sampled_mean': safe_mean(stats_by_setting[(constant, pos_thr, neg_thr)]['num_small_pos_sampled']),
                'pos_overlap_assigned_mean': safe_mean(stats_by_setting[(constant, pos_thr, neg_thr)]['pos_overlap_assigned_mean']),
                'pos_overlap_sampled_mean': safe_mean(stats_by_setting[(constant, pos_thr, neg_thr)]['pos_overlap_sampled_mean']),
            },
            'evidence_3_rcnn_input_quality': {
                'small_gt_recall@1000_iou0.5': safe_mean(proposal_ar1000_small),
                'proposal_noise_ratio_iou<0.3': safe_mean(proposal_noise_ratio_03),
                'proposal_noise_ratio_iou<0.5': safe_mean(proposal_noise_ratio_05),
                'proposal_match_ratio_iou>=0.5': safe_mean(proposal_match_ratio_05),
            },
        }
        results.append(item)

    md_lines = []
    md_lines.append('# RPN NWD Threshold Sweep')
    md_lines.append('')
    md_lines.append(f'- Config: {args.config}')
    md_lines.append(f'- Checkpoint: {args.checkpoint}')
    md_lines.append(f'- Num images: {total_images}')
    md_lines.append(f'- Baseline RPN sampler cap: num=256, pos_fraction=0.5 (max positive samples ~= 128)')
    md_lines.append('')
    md_lines.append('## Sweep Table')
    md_lines.append('')
    md_lines.append('| constant | pos_thr | neg_thr | num_pos_assigned_mean | num_pos_sampled_mean | small_pos_assigned_mean | small_pos_sampled_mean | small_gt_recall@1000 | noise<0.5 |')
    md_lines.append('|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for item in results:
        pos_dist = item['evidence_2_rpn_pos_distribution']
        quality = item['evidence_3_rcnn_input_quality']
        md_lines.append(
            f"| {item['constant']:.1f} | {item['pos_iou_thr']:.2f} | {item['neg_iou_thr']:.2f} | "
            f"{pos_dist['num_pos_assigned_mean']:.3f} | {pos_dist['num_pos_sampled_mean']:.3f} | "
            f"{pos_dist['num_small_pos_assigned_mean']:.3f} | {pos_dist['num_small_pos_sampled_mean']:.3f} | "
            f"{quality['small_gt_recall@1000_iou0.5']:.6f} | {quality['proposal_noise_ratio_iou<0.5']:.6f} |")

    md_lines.append('')
    md_lines.append('## Fixed Proposal Metrics for This Checkpoint')
    md_lines.append('')
    md_lines.append('| AR@100_all | AR@1000_all | AR@100_small | AR@1000_small | noise<0.3 | noise<0.5 | match>=0.5 |')
    md_lines.append('|---:|---:|---:|---:|---:|---:|---:|')
    md_lines.append(
        f"| {safe_mean(proposal_ar100_all):.6f} | {safe_mean(proposal_ar1000_all):.6f} | "
        f"{safe_mean(proposal_ar100_small):.6f} | {safe_mean(proposal_ar1000_small):.6f} | "
        f"{safe_mean(proposal_noise_ratio_03):.6f} | {safe_mean(proposal_noise_ratio_05):.6f} | "
        f"{safe_mean(proposal_match_ratio_05):.6f} |")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(md_lines), encoding='utf-8')

    if args.json_out is None:
        json_out = out_path.with_suffix('.json')
    else:
        json_out = Path(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    with json_out.open('w', encoding='utf-8') as handle:
        json.dump({
            'config': args.config,
            'checkpoint': args.checkpoint,
            'num_images': total_images,
            'results': results,
            'fixed_proposal_metrics': {
                'AR@100_all': safe_mean(proposal_ar100_all),
                'AR@1000_all': safe_mean(proposal_ar1000_all),
                'AR@100_small': safe_mean(proposal_ar100_small),
                'AR@1000_small': safe_mean(proposal_ar1000_small),
                'proposal_noise_ratio_iou<0.3': safe_mean(proposal_noise_ratio_03),
                'proposal_noise_ratio_iou<0.5': safe_mean(proposal_noise_ratio_05),
                'proposal_match_ratio_iou>=0.5': safe_mean(proposal_match_ratio_05),
            }
        }, handle, indent=2, ensure_ascii=False)

    print('\n'.join(md_lines))
    print(f'Saved markdown: {out_path}')
    print(f'Saved json: {json_out}')


if __name__ == '__main__':
    main()
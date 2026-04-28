#!/usr/bin/env python3
"""Analyze whether RPN-side NWD helps dense tiny-object detection chain.

This script outputs three evidence groups for a given checkpoint:
1) RPN proposal recall (AR@100/AR@1000, all + small)
2) RPN positive assignment statistics
3) RCNN-input proposal quality (small recall + noise ratio)
"""

from __future__ import annotations

import argparse
import copy
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
from mmdet.registry import DATASETS
from mmdet.structures.bbox import bbox_overlaps, cat_boxes, get_box_tensor
from mmdet.utils import register_all_modules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='RPN/NWD chain analyzer')
    parser.add_argument('config', help='Config path')
    parser.add_argument('checkpoint', help='Checkpoint path')
    parser.add_argument('--device', default='cuda:0', help='Device')
    parser.add_argument('--max-images', type=int, default=0,
                        help='Max images to analyze; 0 means all val images')
    parser.add_argument('--out', required=True, help='Output JSON path')
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

    ious = bbox_overlaps(topk_props, gt_bboxes)  # [K, G]
    if ious.numel() == 0:
        return 0.0

    gt_max = ious.max(dim=0).values
    recalls = [float((gt_max >= thr).float().mean().item())
               for thr in iou_thresholds]
    return float(sum(recalls) / len(recalls))


def collect_assigner_stats(model,
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
            'pos_overlap_assigned_mean': float('nan'),
            'pos_overlap_assigned_var': float('nan'),
            'pos_overlap_sampled_mean': float('nan'),
            'pos_overlap_sampled_var': float('nan'),
        }

    priors = flat_anchors[inside_flags]
    pred_instances = InstanceData(priors=priors)
    assign_result = model.rpn_head.assigner.assign(
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

    assigned_overlaps = (
        assign_result.max_overlaps[assigned_pos_inds]
        if assigned_pos_inds.numel() > 0
        else torch.empty(0, device=priors.device))
    sampled_overlaps = (
        assign_result.max_overlaps[sampled_pos_inds]
        if sampled_pos_inds.numel() > 0
        else torch.empty(0, device=priors.device))

    return {
        'num_pos_assigned': float(assigned_pos_inds.numel()),
        'num_pos_sampled': float(sampled_pos_inds.numel()),
        'pos_overlap_assigned_mean': (
            float(assigned_overlaps.mean().item())
            if assigned_overlaps.numel() > 0 else float('nan')),
        'pos_overlap_assigned_var': (
            float(assigned_overlaps.var(unbiased=False).item())
            if assigned_overlaps.numel() > 1 else 0.0),
        'pos_overlap_sampled_mean': (
            float(sampled_overlaps.mean().item())
            if sampled_overlaps.numel() > 0 else float('nan')),
        'pos_overlap_sampled_var': (
            float(sampled_overlaps.var(unbiased=False).item())
            if sampled_overlaps.numel() > 1 else 0.0),
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

    iou_thresholds = [x / 100 for x in range(50, 100, 5)]
    small_thr = 32.0 * 32.0

    ar100_all = []
    ar1000_all = []
    ar100_small = []
    ar1000_small = []

    pos_assigned = []
    pos_sampled = []
    ov_assigned_mean = []
    ov_assigned_var = []
    ov_sampled_mean = []
    ov_sampled_var = []

    rcnn_small_recall_1000 = []
    rcnn_noise_ratio_03 = []
    rcnn_noise_ratio_05 = []
    rcnn_match_ratio_05 = []

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
            small_mask = gt_areas < small_thr if gt_areas.numel() > 0 else torch.empty(0, dtype=torch.bool, device=proposal_bboxes.device)
            gt_bboxes_small = gt_bboxes[small_mask] if gt_bboxes.numel() > 0 else gt_bboxes

            top100 = proposal_bboxes[:100]
            top1000 = proposal_bboxes[:1000]

            ar100_all.append(compute_ar(top100, gt_bboxes, iou_thresholds))
            ar1000_all.append(compute_ar(top1000, gt_bboxes, iou_thresholds))
            ar100_small.append(compute_ar(top100, gt_bboxes_small, iou_thresholds))
            ar1000_small.append(compute_ar(top1000, gt_bboxes_small, iou_thresholds))

            assign_stats = collect_assigner_stats(model, cls_scores, batch_data_sample)
            pos_assigned.append(assign_stats['num_pos_assigned'])
            pos_sampled.append(assign_stats['num_pos_sampled'])
            ov_assigned_mean.append(assign_stats['pos_overlap_assigned_mean'])
            ov_assigned_var.append(assign_stats['pos_overlap_assigned_var'])
            ov_sampled_mean.append(assign_stats['pos_overlap_sampled_mean'])
            ov_sampled_var.append(assign_stats['pos_overlap_sampled_var'])

            if gt_bboxes.numel() == 0:
                rcnn_small_recall_1000.append(float('nan'))
                rcnn_noise_ratio_03.append(1.0 if top1000.numel() > 0 else float('nan'))
                rcnn_noise_ratio_05.append(1.0 if top1000.numel() > 0 else float('nan'))
                rcnn_match_ratio_05.append(0.0 if top1000.numel() > 0 else float('nan'))
            elif top1000.numel() == 0:
                rcnn_small_recall_1000.append(0.0)
                rcnn_noise_ratio_03.append(float('nan'))
                rcnn_noise_ratio_05.append(float('nan'))
                rcnn_match_ratio_05.append(float('nan'))
            else:
                ious = bbox_overlaps(top1000, gt_bboxes)
                max_iou_per_prop = ious.max(dim=1).values
                rcnn_noise_ratio_03.append(float((max_iou_per_prop < 0.3).float().mean().item()))
                rcnn_noise_ratio_05.append(float((max_iou_per_prop < 0.5).float().mean().item()))
                rcnn_match_ratio_05.append(float((max_iou_per_prop >= 0.5).float().mean().item()))

                if gt_bboxes_small.numel() == 0:
                    rcnn_small_recall_1000.append(float('nan'))
                else:
                    ious_small = bbox_overlaps(top1000, gt_bboxes_small)
                    max_iou_per_small_gt = ious_small.max(dim=0).values
                    rcnn_small_recall_1000.append(
                        float((max_iou_per_small_gt >= 0.5).float().mean().item()))

            if (idx + 1) % 50 == 0 or (idx + 1) == total_images:
                print(f'[{idx + 1}/{total_images}] processed')

    result = {
        'config': str(args.config),
        'checkpoint': str(args.checkpoint),
        'num_images': total_images,
        'evidence_1_rpn_recall': {
            'AR@100_all': safe_mean(ar100_all),
            'AR@1000_all': safe_mean(ar1000_all),
            'AR@100_small': safe_mean(ar100_small),
            'AR@1000_small': safe_mean(ar1000_small),
        },
        'evidence_2_rpn_pos_distribution': {
            'num_pos_assigned_mean': safe_mean(pos_assigned),
            'num_pos_assigned_std': safe_std(pos_assigned),
            'num_pos_sampled_mean': safe_mean(pos_sampled),
            'num_pos_sampled_std': safe_std(pos_sampled),
            'pos_overlap_assigned_mean': safe_mean(ov_assigned_mean),
            'pos_overlap_assigned_var_mean': safe_mean(ov_assigned_var),
            'pos_overlap_sampled_mean': safe_mean(ov_sampled_mean),
            'pos_overlap_sampled_var_mean': safe_mean(ov_sampled_var),
        },
        'evidence_3_rcnn_input_quality': {
            'small_gt_recall@1000_iou0.5': safe_mean(rcnn_small_recall_1000),
            'proposal_noise_ratio_iou<0.3': safe_mean(rcnn_noise_ratio_03),
            'proposal_noise_ratio_iou<0.5': safe_mean(rcnn_noise_ratio_05),
            'proposal_match_ratio_iou>=0.5': safe_mean(rcnn_match_ratio_05),
        }
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()

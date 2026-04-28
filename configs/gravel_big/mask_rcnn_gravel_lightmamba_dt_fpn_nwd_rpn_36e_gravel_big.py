# NWD (Normalized Gaussian Wasserstein Distance) — Phase 1: RPN-only
# Only replaces IoU with NWD in the RPN assigner.
# The RoI Head assigner and regression loss remain unchanged.
#
# Why RPN-only first?
#   NWD's biggest impact is on RPN proposal recall for tiny objects,
#   because IoU degeneracy (IoU=0 for non-overlapping tiny boxes)
#   directly causes "no positive anchor" in MaxIoUAssigner.
#   By only changing RPN, we can isolate this effect cleanly.
#
# IMPORTANT: NWD thresholds are NOT yet calibrated.
# The current pos/neg thresholds (0.7/0.3) are carried over from IoU.
# NWD values are generally higher than IoU for the same box pair,
# so this may result in more positive anchors than intended.
# A threshold sweep should be done after the first run.

_base_ = ['./mask_rcnn_gravel_lightmamba_dt_fpn_v2_36e_gravel_big.py']

custom_imports = dict(
    imports=[
        'mmdet.datasets.transforms.gravel_loading',
        'mmdet.models.necks.dt_fpn',
        'mmdet.models.detectors.rs_ig_mask_rcnn',
        'mmdet.models.task_modules.assigners.nwd_calculator',
    ],
    allow_failed_imports=False)

model = dict(
    # === RPN ONLY: Replace IoU with NWD in assigner ===
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                _delete_=True,
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlaps2D_NWD', constant=8.0),
                # NWD thresholds: inherited from IoU, need sweep later
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1))))

work_dir = 'work_dirs_gravel_big/mask_rcnn_gravel_lightmamba_dt_fpn_nwd_rpn_36e_gravel_big'

# NWD (Normalized Gaussian Wasserstein Distance) — Phase 2: RPN + RCNN
# Replaces IoU with NWD in BOTH RPN and RoI Head assigners.
# Regression loss remains L1/SmoothL1 (NOT NWDLoss — see below).
#
# Why not NWDLoss yet?
#   NWDLoss requires decoded (absolute) bbox coordinates as input,
#   but MMDet's default regression pipeline passes encoded deltas.
#   To use NWDLoss correctly, you must either:
#   (a) Set reg_decoded_bbox=True in the head, or
#   (b) Use the fixed NWDLoss version that handles decode internally.
#   See nwd_loss.py for the corrected implementation.
#
# IMPORTANT: NWD thresholds are NOT yet calibrated.

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
    # === RPN: Replace IoU with NWD in assigner ===
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                _delete_=True,
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlaps2D_NWD', constant=8.0),
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1)),
        rcnn=dict(
            assigner=dict(
                _delete_=True,
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlaps2D_NWD', constant=8.0),
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1))))

work_dir = 'work_dirs_gravel_big/mask_rcnn_gravel_lightmamba_dt_fpn_nwd_rpn_rcnn_36e_gravel_big'

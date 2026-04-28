# NWD (Normalized Gaussian Wasserstein Distance) — RPN v2
# Calibrated from the sweep result to restore RPN positive sample counts
# to a baseline-like range before evaluating downstream impact.

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
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                _delete_=True,
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlaps2D_NWD', constant=8.0),
                pos_iou_thr=0.53,
                neg_iou_thr=0.10,
                min_pos_iou=0.10,
                match_low_quality=True,
                ignore_iof_thr=-1))))

work_dir = 'work_dirs_gravel_big/mask_rcnn_gravel_lightmamba_dt_fpn_nwd_rpn_v2_36e_gravel_big'
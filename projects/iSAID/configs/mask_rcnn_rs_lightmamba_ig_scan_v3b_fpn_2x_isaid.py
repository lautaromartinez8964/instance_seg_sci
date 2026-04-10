_base_ = ['./mask_rcnn_rs_lightmamba_ig_scan_v3b_fpn_1x_isaid.py']

custom_imports = dict(
    imports=[
        'mmdet.models.backbones.rs_lightmamba.lightmamba_backbone',
        'mmdet.models.detectors.rs_ig_mask_rcnn',
        'mmdet.engine.hooks.ig_fg_loss_decay_hook'
    ],
    allow_failed_imports=False)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=1)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=24,
        by_epoch=True,
        milestones=[16, 22],
        gamma=0.1)
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=3,
        save_best='coco/segm_mAP',
        rule='greater'))

custom_hooks = [
    dict(
        type='IGFGLinearDecayHook',
        start_weight=0.2,
        end_weight=0.05,
        begin_epoch=8,
        end_epoch=24)
]

work_dir = 'work_dirs/mask_rcnn_rs_lightmamba_ig_scan_v3b_fpn_2x_isaid'
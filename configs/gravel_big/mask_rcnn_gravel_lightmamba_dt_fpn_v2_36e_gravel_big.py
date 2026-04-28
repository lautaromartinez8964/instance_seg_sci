_base_ = ['./mask_rcnn_rs_lightmamba_fpn_36e_gravel_big.py']

custom_imports = dict(
    imports=[
        'mmdet.datasets.transforms.gravel_loading',
        'mmdet.models.necks.dt_fpn',
        'mmdet.models.detectors.rs_ig_mask_rcnn'
    ],
    allow_failed_imports=False)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='LoadGravelDistanceTransform',
        distance_root='data/gravel_big_mmdet/auxiliary_labels_coreband'),
    dict(
        type='RandomChoiceResize',
        scales=[(512, 512), (640, 640), (704, 704)],
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackDetInputs')
]

model = dict(
    type='RSMaskRCNN',
    neck=dict(
        _delete_=True,
        type='DTFPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5,
        guided_levels=[0, 1, 2],
        dt_head_channels=128,
        dt_loss_weight=0.2))

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0002,
        weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(batch_size=1, num_workers=2, persistent_workers=True)
test_dataloader = dict(batch_size=1, num_workers=2, persistent_workers=True)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=10),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=40,
        save_best='coco/segm_mAP',
        rule='greater'))

work_dir = 'work_dirs_gravel_big/mask_rcnn_gravel_lightmamba_dt_fpn_v2_36e_gravel_big'
_base_ = ['./rtmdet_ins_tiny_baseline_120e_gravel_big.py']


custom_imports = dict(
    imports=[
        'mmdet.datasets.transforms.gravel_loading',
        'mmdet.models.necks.dt_cspnext_pafpn',
        'mmdet.models.detectors.rtmdet_aux'
    ],
    allow_failed_imports=False)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, poly2mask=False),
    dict(
        type='LoadGravelDistanceTransform',
        distance_root='data/gravel_big_mmdet/auxiliary_labels_coreband'),
    dict(
        type='RandomResize',
        scale=(640, 640),
        ratio_range=(0.5, 2.0),
        resize_type='Resize',
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=(640, 640),
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='Pad',
        size=(640, 640),
        pad_val=dict(img=(114, 114, 114), seg=255)),
    dict(type='PackDetInputs')
]

model = dict(
    type='RTMDetWithAuxNeck',
    neck=dict(
        _delete_=True,
        type='DTCSPNeXtPAFPN',
        in_channels=[96, 192, 384],
        out_channels=96,
        num_csp_blocks=1,
        expand_ratio=0.5,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True),
        guided_levels=[0, 1],
        dt_mode='shared',
        dt_decoder_source='topdown',
        dt_head_channels=128,
        use_residual_fusion=True,
        use_channel_attention=False),
    bbox_head=dict(in_channels=96, feat_channels=96))

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49)
]

work_dir = 'work_dirs_gravel_big/rtmdet_ins_tiny_shared_dt_fpn_v2c_120e_gravel_big'
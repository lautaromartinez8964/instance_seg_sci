albu_train_transforms = [
    dict(
        p=0.7,
        transforms=[
            dict(p=1.0, type='VerticalFlip'),
            dict(p=1.0, type='RandomRotate90'),
            dict(
                interpolation=1,
                p=1.0,
                rotate_limit=15,
                scale_limit=0.1,
                shift_limit=0.0625,
                type='ShiftScaleRotate'),
        ],
        type='OneOf'),
    dict(
        p=0.6,
        transforms=[
            dict(
                brightness_limit=[
                    0.1,
                    0.25,
                ],
                contrast_limit=[
                    0.1,
                    0.25,
                ],
                p=1.0,
                type='RandomBrightnessContrast'),
            dict(
                hue_shift_limit=12,
                p=1.0,
                sat_shift_limit=20,
                type='HueSaturationValue',
                val_shift_limit=12),
            dict(
                b_shift_limit=10,
                g_shift_limit=10,
                p=1.0,
                r_shift_limit=10,
                type='RGBShift'),
            dict(clip_limit=2.0, p=1.0, type='CLAHE'),
        ],
        type='OneOf'),
    dict(
        p=0.25,
        transforms=[
            dict(
                p=1.0,
                quality_lower=85,
                quality_upper=95,
                type='JpegCompression'),
            dict(blur_limit=3, p=1.0, type='MotionBlur'),
            dict(blur_limit=3, p=1.0, type='MedianBlur'),
            dict(blur_limit=3, p=1.0, type='Blur'),
        ],
        type='OneOf'),
]
auto_scale_lr = dict(base_batch_size=16, enable=False)
backend_args = None
data_root = 'data/gravel_roboflow_414_mmdet/'
dataset_type = 'GravelDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=3,
        rule='greater',
        save_best='coco/segm_mAP',
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
load_from = 'checkpoints/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
metainfo = dict(
    classes=('gravel', ), palette=[
        (
            255,
            140,
            0,
        ),
    ])
model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=1,
        init_cfg=dict(checkpoint='torchvision://resnet50', type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='pytorch',
        type='ResNet'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_mask=True,
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=5,
        out_channels=256,
        type='FPN'),
    roi_head=dict(
        bbox_head=dict(
            bbox_coder=dict(
                target_means=[
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                target_stds=[
                    0.1,
                    0.1,
                    0.2,
                    0.2,
                ],
                type='DeltaXYWHBBoxCoder'),
            fc_out_channels=1024,
            in_channels=256,
            loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
            loss_cls=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
            num_classes=1,
            reg_class_agnostic=False,
            roi_feat_size=7,
            type='Shared2FCBBoxHead'),
        bbox_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(output_size=7, sampling_ratio=0, type='RoIAlign'),
            type='SingleRoIExtractor'),
        mask_head=dict(
            conv_out_channels=256,
            in_channels=256,
            loss_mask=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_mask=True),
            num_classes=1,
            num_convs=4,
            type='FCNMaskHead'),
        mask_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(output_size=14, sampling_ratio=0, type='RoIAlign'),
            type='SingleRoIExtractor'),
        type='StandardRoIHead'),
    rpn_head=dict(
        anchor_generator=dict(
            ratios=[
                0.5,
                1.0,
                2.0,
            ],
            scales=[
                8,
            ],
            strides=[
                4,
                8,
                16,
                32,
                64,
            ],
            type='AnchorGenerator'),
        bbox_coder=dict(
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            type='DeltaXYWHBBoxCoder'),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
        loss_cls=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
        type='RPNHead'),
    test_cfg=dict(
        rcnn=dict(
            mask_thr_binary=0.5,
            max_per_img=100,
            nms=dict(iou_threshold=0.5, type='nms'),
            score_thr=0.05),
        rpn=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=1000)),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=True,
                min_pos_iou=0.5,
                neg_iou_thr=0.5,
                pos_iou_thr=0.5,
                type='MaxIoUAssigner'),
            debug=False,
            mask_size=28,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=True,
                neg_pos_ub=-1,
                num=512,
                pos_fraction=0.25,
                type='RandomSampler')),
        rpn=dict(
            allowed_border=-1,
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=True,
                min_pos_iou=0.3,
                neg_iou_thr=0.3,
                pos_iou_thr=0.7,
                type='MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=False,
                neg_pos_ub=-1,
                num=256,
                pos_fraction=0.5,
                type='RandomSampler')),
        rpn_proposal=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=2000)),
    type='MaskRCNN')
optim_wrapper = dict(
    optimizer=dict(lr=0.0025, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=100, start_factor=0.001, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=36,
        gamma=0.1,
        milestones=[
            24,
            33,
        ],
        type='MultiStepLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/instances_test.json',
        backend_args=None,
        data_prefix=dict(img='test/'),
        data_root='data/gravel_roboflow_414_mmdet/',
        metainfo=dict(classes=('gravel', ), palette=[
            (
                255,
                140,
                0,
            ),
        ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                640,
                640,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='GravelDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=False,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='data/gravel_roboflow_414_mmdet/annotations/instances_test.json',
    backend_args=None,
    format_only=False,
    metric='segm',
    type='FasterCocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        640,
        640,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(max_epochs=36, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=2,
    dataset=dict(
        ann_file='annotations/instances_train.json',
        backend_args=None,
        data_prefix=dict(img='train/'),
        data_root='data/gravel_roboflow_414_mmdet/',
        filter_cfg=dict(filter_empty_gt=False, min_size=1),
        metainfo=dict(classes=('gravel', ), palette=[
            (
                255,
                140,
                0,
            ),
        ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                keep_ratio=True,
                scales=[
                    (
                        480,
                        480,
                    ),
                    (
                        576,
                        576,
                    ),
                    (
                        640,
                        640,
                    ),
                    (
                        704,
                        704,
                    ),
                    (
                        800,
                        800,
                    ),
                ],
                type='RandomChoiceResize'),
            dict(
                bbox_params=dict(
                    filter_lost_elements=True,
                    format='pascal_voc',
                    label_fields=[
                        'gt_bboxes_labels',
                        'gt_ignore_flags',
                    ],
                    min_visibility=0.0,
                    type='BboxParams'),
                keymap=dict(gt_bboxes='bboxes', gt_masks='masks', img='image'),
                skip_img_without_anno=True,
                transforms=[
                    dict(
                        p=0.7,
                        transforms=[
                            dict(p=1.0, type='VerticalFlip'),
                            dict(p=1.0, type='RandomRotate90'),
                            dict(
                                interpolation=1,
                                p=1.0,
                                rotate_limit=15,
                                scale_limit=0.1,
                                shift_limit=0.0625,
                                type='ShiftScaleRotate'),
                        ],
                        type='OneOf'),
                    dict(
                        p=0.6,
                        transforms=[
                            dict(
                                brightness_limit=[
                                    0.1,
                                    0.25,
                                ],
                                contrast_limit=[
                                    0.1,
                                    0.25,
                                ],
                                p=1.0,
                                type='RandomBrightnessContrast'),
                            dict(
                                hue_shift_limit=12,
                                p=1.0,
                                sat_shift_limit=20,
                                type='HueSaturationValue',
                                val_shift_limit=12),
                            dict(
                                b_shift_limit=10,
                                g_shift_limit=10,
                                p=1.0,
                                r_shift_limit=10,
                                type='RGBShift'),
                            dict(clip_limit=2.0, p=1.0, type='CLAHE'),
                        ],
                        type='OneOf'),
                    dict(
                        p=0.25,
                        transforms=[
                            dict(
                                p=1.0,
                                quality_lower=85,
                                quality_upper=95,
                                type='JpegCompression'),
                            dict(blur_limit=3, p=1.0, type='MotionBlur'),
                            dict(blur_limit=3, p=1.0, type='MedianBlur'),
                            dict(blur_limit=3, p=1.0, type='Blur'),
                        ],
                        type='OneOf'),
                ],
                type='Albu'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackDetInputs'),
        ],
        type='GravelDataset'),
    num_workers=2,
    persistent_workers=False,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        keep_ratio=True,
        scales=[
            (
                480,
                480,
            ),
            (
                576,
                576,
            ),
            (
                640,
                640,
            ),
            (
                704,
                704,
            ),
            (
                800,
                800,
            ),
        ],
        type='RandomChoiceResize'),
    dict(
        bbox_params=dict(
            filter_lost_elements=True,
            format='pascal_voc',
            label_fields=[
                'gt_bboxes_labels',
                'gt_ignore_flags',
            ],
            min_visibility=0.0,
            type='BboxParams'),
        keymap=dict(gt_bboxes='bboxes', gt_masks='masks', img='image'),
        skip_img_without_anno=True,
        transforms=[
            dict(
                p=0.7,
                transforms=[
                    dict(p=1.0, type='VerticalFlip'),
                    dict(p=1.0, type='RandomRotate90'),
                    dict(
                        interpolation=1,
                        p=1.0,
                        rotate_limit=15,
                        scale_limit=0.1,
                        shift_limit=0.0625,
                        type='ShiftScaleRotate'),
                ],
                type='OneOf'),
            dict(
                p=0.6,
                transforms=[
                    dict(
                        brightness_limit=[
                            0.1,
                            0.25,
                        ],
                        contrast_limit=[
                            0.1,
                            0.25,
                        ],
                        p=1.0,
                        type='RandomBrightnessContrast'),
                    dict(
                        hue_shift_limit=12,
                        p=1.0,
                        sat_shift_limit=20,
                        type='HueSaturationValue',
                        val_shift_limit=12),
                    dict(
                        b_shift_limit=10,
                        g_shift_limit=10,
                        p=1.0,
                        r_shift_limit=10,
                        type='RGBShift'),
                    dict(clip_limit=2.0, p=1.0, type='CLAHE'),
                ],
                type='OneOf'),
            dict(
                p=0.25,
                transforms=[
                    dict(
                        p=1.0,
                        quality_lower=85,
                        quality_upper=95,
                        type='JpegCompression'),
                    dict(blur_limit=3, p=1.0, type='MotionBlur'),
                    dict(blur_limit=3, p=1.0, type='MedianBlur'),
                    dict(blur_limit=3, p=1.0, type='Blur'),
                ],
                type='OneOf'),
        ],
        type='Albu'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/instances_val.json',
        backend_args=None,
        data_prefix=dict(img='val/'),
        data_root='data/gravel_roboflow_414_mmdet/',
        metainfo=dict(classes=('gravel', ), palette=[
            (
                255,
                140,
                0,
            ),
        ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                640,
                640,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='GravelDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=False,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='data/gravel_roboflow_414_mmdet/annotations/instances_val.json',
    backend_args=None,
    format_only=False,
    metric='segm',
    type='FasterCocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'work_dirs_gravel/mask_rcnn_r50_fpn_36e_gravel_414_augment'

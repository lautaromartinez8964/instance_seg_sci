_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/gravel_big_instance.py'
]

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)))

train_dataloader = dict(batch_size=4, num_workers=4, persistent_workers=True)
val_dataloader = dict(batch_size=1, num_workers=2, persistent_workers=True)
test_dataloader = dict(batch_size=1, num_workers=2, persistent_workers=True)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=100),
    dict(
        type='MultiStepLR',
        begin=0,
        end=24,
        by_epoch=True,
        milestones=[16, 22],
        gamma=0.1)
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001))

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=3,
        save_best='coco/segm_mAP',
        rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

val_evaluator = dict(
    type='FasterCocoMetric',
    ann_file='data/gravel_big_mmdet/annotations/instances_val.json',
    metric='segm',
    format_only=False,
    backend_args=None)

test_evaluator = dict(
    type='FasterCocoMetric',
    ann_file='data/gravel_big_mmdet/annotations/instances_test.json',
    metric='segm',
    format_only=False,
    backend_args=None)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'

auto_scale_lr = dict(enable=False, base_batch_size=16)

load_from = 'checkpoints/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'
resume = False

work_dir = 'work_dirs_gravel_big/mask_rcnn_r50_fpn_24e_gravel_big'
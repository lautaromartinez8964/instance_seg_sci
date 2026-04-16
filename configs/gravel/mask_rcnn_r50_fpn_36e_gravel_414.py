_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/gravel_instance_414.py'
]

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)))

# 单卡保守基线：batch_size=2，对应 lr=0.02 * (2 / 16) = 0.0025
train_dataloader = dict(batch_size=2, num_workers=2, persistent_workers=False)
val_dataloader = dict(batch_size=1, num_workers=1, persistent_workers=False)
test_dataloader = dict(batch_size=1, num_workers=1, persistent_workers=False)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=36, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=100),
    dict(
        type='MultiStepLR',
        begin=0,
        end=36,
        by_epoch=True,
        milestones=[24, 33],
        gamma=0.1)
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001))

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
    ann_file='data/gravel_roboflow_414_mmdet/annotations/instances_val.json',
    metric='segm',
    format_only=False,
    backend_args=None)

test_evaluator = dict(
    type='FasterCocoMetric',
    ann_file='data/gravel_roboflow_414_mmdet/annotations/instances_test.json',
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

work_dir = 'work_dirs_gravel/mask_rcnn_r50_fpn_36e_gravel_414'
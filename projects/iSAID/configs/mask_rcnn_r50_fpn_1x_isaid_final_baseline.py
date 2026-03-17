_base_ = [
    '../../../configs/_base_/models/mask-rcnn_r50_fpn.py',
    '../../../configs/_base_/datasets/isaid_instance.py',
    '../../../configs/_base_/schedules/schedule_1x.py',
    '../../../configs/_base_/default_runtime.py'
]

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=15),
        mask_head=dict(num_classes=15)
    )
)

# 单卡 batch=4（你5090可尝试），先稳一点
train_dataloader = dict(batch_size=4, num_workers=2, persistent_workers=False)
val_dataloader = dict(batch_size=1, num_workers=0, persistent_workers=False)
test_dataloader = dict(batch_size=1, num_workers=0, persistent_workers=False)

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
)

# 先跑2x更稳，给出合理baseline
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=1)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3)
)
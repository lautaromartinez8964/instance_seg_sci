_base_ = './mask_rcnn_r50_fpn_1x_isaid.py'

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=2, val_interval=1)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=2)
)
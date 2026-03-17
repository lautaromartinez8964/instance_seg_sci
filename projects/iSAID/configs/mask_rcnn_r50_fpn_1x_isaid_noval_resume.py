_base_ = './mask_rcnn_r50_fpn_1x_isaid.py'

# 仍然训练到12 epoch（按你原计划）
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1000)

train_dataloader = dict(num_workers=0, persistent_workers=False)
val_dataloader = dict(num_workers=0, persistent_workers=False)
test_dataloader = dict(num_workers=0, persistent_workers=False)

env_cfg = dict(mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0))
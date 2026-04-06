_base_ = ['./mask_rcnn_rs_lightmamba_ig_scan_v5b_dt_fpn_1x_isaid.py']

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=2)

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

work_dir = 'work_dirs/mask_rcnn_rs_lightmamba_ig_scan_v5b_dt_fpn_2x_isaid'
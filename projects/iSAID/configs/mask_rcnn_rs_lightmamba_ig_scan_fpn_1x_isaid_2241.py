_base_ = ['./mask_rcnn_rs_lightmamba_ig_scan_fpn_1x_isaid.py']

model = dict(
    backbone=dict(
        depths=[2, 2, 4, 1],
        dims=[48, 96, 192, 384]),
    neck=dict(
        in_channels=[48, 96, 192, 384]))

work_dir = 'work_dirs/mask_rcnn_rs_lightmamba_ig_scan_fpn_1x_isaid_2241'
_base_ = ['./mask_rcnn_rs_lightmamba_ig_scan_v5a_dt_fpn_1x_isaid.py']

model = dict(
    backbone=dict(
        ig_dt_scale=0.06))

work_dir = 'work_dirs/mask_rcnn_rs_lightmamba_ig_scan_v5b_dt_fpn_1x_isaid'
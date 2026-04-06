_base_ = ['./mask_rcnn_rs_lightmamba_ig_scan_v5b_dt_fpn_1x_isaid.py']

model = dict(
    backbone=dict(
        ig_output_scale=0.05))

work_dir = 'work_dirs/mask_rcnn_rs_lightmamba_ig_scan_v6_dual_mod_fpn_1x_isaid'
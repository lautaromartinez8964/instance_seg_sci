_base_ = ['./mask_rcnn_rs_lightmamba_ig_scan_v3b_fpn_1x_isaid.py']

model = dict(backbone=dict(ig_region_score_mode='avg'))

work_dir = 'work_dirs/mask_rcnn_rs_lightmamba_ig_scan_v3d_fpn_1x_isaid'
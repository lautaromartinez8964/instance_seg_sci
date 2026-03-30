_base_ = ['./mask_rcnn_rs_lightmamba_fpn_1x_isaid.py']

model = dict(
    backbone=dict(
        ig_scan_stages=[3],
        ig_region_size=4,
        ig_guidance_scale=0.1,
        ig_lk_size=7,
        ig_descending_only=True))

work_dir = 'work_dirs/mask_rcnn_rs_lightmamba_ig_scan_v2_fpn_1x_isaid'
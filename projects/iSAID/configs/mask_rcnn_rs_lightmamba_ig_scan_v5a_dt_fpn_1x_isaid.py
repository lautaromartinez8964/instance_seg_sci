_base_ = ['./mask_rcnn_rs_lightmamba_fpn_1x_isaid.py']

model = dict(
    backbone=dict(
        forward_type='v05_noz',
        ig_scan_stages=[2, 3],
        ig_mode='dt_gate',
        ig_dt_scale=0.03,
        ig_lk_size=7,
        ig_fg_norm_type='gn',
        ig_fg_gn_groups=8,
        ig_use_fg_loss=True,
        ig_fg_loss_weight=0.2))

work_dir = 'work_dirs/mask_rcnn_rs_lightmamba_ig_scan_v5a_dt_fpn_1x_isaid'
_base_ = ['./mask_rcnn_rs_lightmamba_fpn_1x_isaid.py']

model = dict(
    backbone=dict(
        forward_type='v05',
        ig_scan_stages=[2, 3],
        ig_mode='z_gate',
        ig_gate_mode='bidirectional',
        ig_gate_scale=0.1,
        ig_lk_size=7,
        ig_fg_norm_type='gn',
        ig_fg_gn_groups=8,
        ig_use_fg_loss=True,
        ig_fg_loss_weight=0.2))

work_dir = 'work_dirs/mask_rcnn_rs_lightmamba_ig_scan_v4b_zgate_fpn_1x_isaid'
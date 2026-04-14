_base_ = ['./mask_rcnn_rs_lightmamba_s4_global_attn_ig_fpn_1x_isaid.py']

model = dict(
    backbone=dict(
        guidance_norm_type='gn',
        guidance_gn_groups=8),
    neck=dict(guided_levels=[0]))

work_dir = 'work_dirs/mask_rcnn_rs_lightmamba_s4_global_attn_ig_fpn_v2_1x_isaid'

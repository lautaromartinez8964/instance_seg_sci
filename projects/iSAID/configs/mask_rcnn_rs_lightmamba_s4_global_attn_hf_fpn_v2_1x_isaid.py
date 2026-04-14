_base_ = ['./mask_rcnn_rs_lightmamba_s4_global_attn_hf_fpn_1x_isaid.py']

model = dict(
    backbone=dict(attention_use_pos_embed=False),
    neck=dict(guided_levels=[0, 1]))

work_dir = 'work_dirs/mask_rcnn_rs_lightmamba_s4_global_attn_hf_fpn_v2_1x_isaid'
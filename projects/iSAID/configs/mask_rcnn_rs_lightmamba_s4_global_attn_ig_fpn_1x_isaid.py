_base_ = ['./mask_rcnn_rs_lightmamba_s4_global_attn_fpn_1x_isaid.py']

model = dict(
    backbone=dict(
        attention_fg_stage=None,
        attention_use_fg_loss=False,
        output_guidance_map=True,
        guidance_stages=[2, 3],
        guidance_hidden_dim=256,
        guidance_use_fg_loss=True,
        guidance_loss_weight=0.2,
        guidance_lk_size=7,
        guidance_norm_type='bn'),
    neck=dict(
        _delete_=True,
        type='IG_FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5,
        guided_levels=[0, 1]))

work_dir = 'work_dirs/mask_rcnn_rs_lightmamba_s4_global_attn_ig_fpn_1x_isaid'
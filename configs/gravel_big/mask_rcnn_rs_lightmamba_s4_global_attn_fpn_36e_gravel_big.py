_base_ = ['./mask_rcnn_rs_lightmamba_fpn_36e_gravel_big.py']

model = dict(
    backbone=dict(
        attention_stages=[3],
        attention_num_heads=8,
        attention_mlp_ratio=4.0,
        attention_qkv_bias=True,
        attention_attn_drop=0.0,
        attention_proj_drop=0.0,
        attention_fg_stage=3,
        attention_use_fg_loss=True,
        attention_fg_loss_weight=0.2,
        attention_fg_lk_size=7,
        attention_fg_norm_type='bn'))

work_dir = 'work_dirs_gravel_big/mask_rcnn_rs_lightmamba_s4_global_attn_fpn_36e_gravel_big'
_base_ = ['./mask_rcnn_rs_lightmamba_s4_global_attn_fpn_36e_gravel_big.py']

model = dict(
    backbone=dict(
        output_hf_maps=True,
        hf_map_stages=[0, 1, 2]),
    neck=dict(
        _delete_=True,
        type='HF_FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5))

work_dir = 'work_dirs_gravel_big/mask_rcnn_rs_lightmamba_s4_global_attn_hf_fpn_36e_gravel_big'
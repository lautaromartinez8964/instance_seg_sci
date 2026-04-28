_base_ = ['./mask_rcnn_gravel_lightmamba_dt_fpn_v2_36e_gravel_big.py']

model = dict(
    neck=dict(
        dt_mode='shared_refined',
        dt_head_channels=128,
        dt_loss_weight=0.2))

work_dir = 'work_dirs_gravel_big/mask_rcnn_gravel_lightmamba_dt_fpn_v3b_36e_gravel_big'
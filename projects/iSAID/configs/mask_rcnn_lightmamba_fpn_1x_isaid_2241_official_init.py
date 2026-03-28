_base_ = ['./mask_rcnn_lightmamba_fpn_1x_isaid.py']

model = dict(
    backbone=dict(
        depths=[2, 2, 4, 1],
        pretrained='checkpoints/vmamba/vssm_tiny_0230_ckpt_epoch_262.pth',
        pretrained_adapter='official_vmamba'))

work_dir = 'work_dirs/mask_rcnn_lightmamba_fpn_1x_isaid_2241_official_init'
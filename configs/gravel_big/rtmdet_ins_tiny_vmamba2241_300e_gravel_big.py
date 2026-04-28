_base_ = ['./rtmdet_ins_tiny_baseline_300e_gravel_big.py']

custom_imports = dict(
    imports=['mmdet.models.backbones.rtmdet_vmamba_backbone'],
    allow_failed_imports=False)

model = dict(
    backbone=dict(
        _delete_=True,
        type='RTMDetMMVMambaBackbone',
        out_indices=(1, 2, 3),
        depths=[2, 2, 4, 1],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.2,
        ssm_d_state=1,
        ssm_ratio=2.0,
        ssm_dt_rank='auto',
        ssm_act_layer='silu',
        ssm_conv=3,
        ssm_conv_bias=False,
        ssm_drop_rate=0.0,
        ssm_init='v0',
        forward_type='v05_noz',
        mlp_ratio=4.0,
        mlp_act_layer='gelu',
        mlp_drop_rate=0.0,
        gmlp=False,
        patch_norm=True,
        norm_layer='ln',
        downsample_version='v3',
        patchembed_version='v2',
        pretrained='checkpoints/vmamba/vssm_tiny_0230_ckpt_epoch_262.pth'),
    neck=dict(in_channels=[192, 384, 768], out_channels=128, num_csp_blocks=1),
    bbox_head=dict(in_channels=128, feat_channels=128))

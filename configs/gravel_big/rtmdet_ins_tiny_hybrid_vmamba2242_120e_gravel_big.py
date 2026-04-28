_base_ = ['./rtmdet_ins_tiny_baseline_120e_gravel_big.py']

custom_imports = dict(
    imports=['mmdet.models.backbones.rtmdet_vmamba_backbone'],
    allow_failed_imports=False)

model = dict(
    backbone=dict(
        _delete_=True,
        type='RTMDetHybridVMambaBackbone',
        csp_deepen_factor=0.167,
        csp_widen_factor=0.375,
        csp_expand_ratio=0.5,
        csp_channel_attention=True,
        csp_norm_cfg=dict(type='SyncBN'),
        csp_act_cfg=dict(type='SiLU', inplace=True),
        csp_init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint='https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth'),
        vmamba_pretrained='checkpoints/vmamba/vssm_tiny_0230_ckpt_epoch_262.pth',
        vmamba_depths=[2, 2, 4, 2],
        vmamba_dims=[32, 64, 144, 192],
        vmamba_drop_path_rate=0.2,
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
        patchembed_version='v2'),
    neck=dict(in_channels=[96, 144, 192], out_channels=96, num_csp_blocks=1),
    bbox_head=dict(in_channels=96, feat_channels=96))

work_dir = 'work_dirs_gravel_big/rtmdet_ins_tiny_hybrid_vmamba2242_120e_gravel_big'
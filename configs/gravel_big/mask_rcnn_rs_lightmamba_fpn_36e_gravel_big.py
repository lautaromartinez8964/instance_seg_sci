_base_ = ['./mask_rcnn_r50_fpn_36e_gravel_big.py']

custom_imports = dict(
    imports=[
        'mmdet.models.backbones.rs_lightmamba.lightmamba_backbone',
        'mmdet.models.detectors.rs_ig_mask_rcnn'
    ],
    allow_failed_imports=False)

model = dict(
    type='RSMaskRCNN',
    backbone=dict(
        _delete_=True,
        type='RSLightMambaBackbone',
        depths=[2, 2, 9, 2],
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
        official_pretrained='checkpoints/vmamba/vssm_tiny_0230_ckpt_epoch_262.pth',
        pretrained_key='model',
        strict_pretrained=False),
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5))

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.0002,
        weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

load_from = None

work_dir = 'work_dirs_gravel_big/mask_rcnn_rs_lightmamba_fpn_36e_gravel_big'
_base_ = ['./mask_rcnn_rs_lightmamba_fpn_36e_gravel_big.py']

custom_imports = dict(
    imports=[
        'mmdet.datasets.transforms.gravel_loading',
        'mmdet.models.backbones.gravel_lightmamba.ibd_lightmamba_backbone',
        'mmdet.models.detectors.rs_ig_mask_rcnn'
    ],
    allow_failed_imports=False)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='LoadGravelBoundaryMap',
        boundary_root='data/gravel_big_mmdet/auxiliary_labels_coreband',
        load_core_band=True),
    dict(
        type='RandomChoiceResize',
        scales=[(512, 512), (640, 640), (768, 768)],
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackDetInputs')
]

model = dict(
    backbone=dict(
        _delete_=True,
        type='GravelLightMambaIBDBackbone',
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
        strict_pretrained=False,
        ibd_stages=(0, 1),
        ibd_hidden_dim=128,
        ibd_loss_weight=1.0,
        ibd_bce_weight=1.0,
        ibd_dice_weight=1.0,
        ibd_max_pos_weight=4.0,
        ibd_band_loss_weight=0.35,
        ibd_far_neg_weight=2.0,
        bcra_stages=()),
    type='RSMaskRCNN')

train_dataloader = dict(
    num_workers=0,
    persistent_workers=False,
    dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(num_workers=0, persistent_workers=False)
test_dataloader = dict(num_workers=0, persistent_workers=False)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=40,
        save_best='coco/segm_mAP',
        rule='greater'))

work_dir = 'work_dirs_gravel_big/mask_rcnn_gravel_lightmamba_ibd_coreband_fpn_36e_gravel_big'
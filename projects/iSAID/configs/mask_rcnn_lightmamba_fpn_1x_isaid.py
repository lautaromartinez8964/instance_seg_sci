_base_ = [
    '../../../configs/_base_/models/mask-rcnn_r50_fpn.py',
    '../../../configs/_base_/datasets/isaid_instance.py',
    '../../../configs/_base_/schedules/schedule_1x.py',
    '../../../configs/_base_/default_runtime.py'
]

custom_imports = dict(imports=['mmdet.models.backbones.custom_Mamba.vmamba_backbone'], allow_failed_imports=False)

model = dict(
    backbone=dict(
        _delete_=True,
        type='VMambaBackbone',
        dims=[48, 96, 192, 384],
        depths=[2, 2, 9, 2],
        drop_path_rate=0.2, # 适当的 drop path
        out_indices=(0, 1, 2, 3),
        norm_layer='ln2d',
        channel_first=False
    ),
    neck=dict(
        type='FPN',
        in_channels=[48, 96, 192, 384],
        out_channels=256,
        num_outs=5
    ),
    roi_head=dict(
        bbox_head=dict(num_classes=15),
        mask_head=dict(num_classes=15)
    )
)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.0001,
        weight_decay=0.05
    ),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }
    )
)

# iSAID dataset overrides for training
metainfo = dict(classes=(
    'ship', 'store_tank', 'baseball_diamond',
    'tennis_court', 'basketball_court', 'Ground_Track_Field',
    'Bridge', 'Large_Vehicle', 'Small_Vehicle', 'Helicopter',
    'Swimming_pool', 'Roundabout', 'Soccer_ball_field', 'plane',
    'Harbor'
))

train_dataloader = dict(
    batch_size=2, # 取决于显存大小
    dataset=dict(
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=True)
    )
)
val_dataloader = dict(dataset=dict(metainfo=metainfo))
test_dataloader = dict(dataset=dict(metainfo=metainfo))

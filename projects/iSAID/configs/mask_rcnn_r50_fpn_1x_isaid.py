_base_ = [
    '../../../configs/_base_/models/mask-rcnn_r50_fpn.py',
    '../../../configs/_base_/datasets/isaid_instance.py',
    '../../../configs/_base_/schedules/schedule_1x.py',
    '../../../configs/_base_/default_runtime.py'
]

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=15),
        mask_head=dict(num_classes=15)
    )
)

optim_wrapper = dict(
    optimizer=dict(
        type='SGD',
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0001
    )
)

metainfo = dict(classes=(
    'ship', 'store_tank', 'baseball_diamond',
    'tennis_court', 'basketball_court', 'Ground_Track_Field',
    'Bridge', 'Large_Vehicle', 'Small_Vehicle', 'Helicopter',
    'Swimming_pool', 'Roundabout', 'Soccer_ball_field', 'plane',
    'Harbor'
))

train_dataloader = dict(dataset=dict(
    metainfo=metainfo,
    filter_cfg=dict(filter_empty_gt=True)
))
val_dataloader = dict(dataset=dict(metainfo=metainfo))
test_dataloader = dict(dataset=dict(metainfo=metainfo))

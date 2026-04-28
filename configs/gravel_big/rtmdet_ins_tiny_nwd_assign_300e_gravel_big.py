_base_ = ['./rtmdet_ins_tiny_baseline_300e_gravel_big.py']

custom_imports = dict(
    imports=['mmdet.models.task_modules.assigners.nwd_calculator'],
    allow_failed_imports=False)

model = dict(
    train_cfg=dict(
        assigner=dict(
            type='DynamicSoftLabelAssigner',
            topk=13,
            iou_weight=3.0,
            iou_calculator=dict(type='BboxOverlaps2D_NWD', constant=8.0)),
        allowed_border=-1,
        pos_weight=-1,
        debug=False))

work_dir = 'work_dirs_gravel_big/rtmdet_ins_tiny_nwd_assign_300e_gravel_big'
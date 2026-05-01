"""RTMDet-Ins-Tiny + DT-FPN v3 + UAGD L1 Boundary Supervision (Phase 2).

继承 DT-FPN v3 配置，在最后 20 个 epoch (stage2) 中叠加 SAM2 离线边界监督。

关键改动：
  1. train_pipeline_stage2 加入 LoadSAM2BoundarySupervision
  2. neck 中启用 uagd_boundary_weight=1.0（关闭 dt_loss 防止 gt_sem_seg 语义冲突）
  3. load_from 指向已收敛的 DT-FPN v3 checkpoint
"""

_base_ = ['./rtmdet_ins_tiny_shared_dt_fpn_v3_120e_gravel_big.py']

custom_imports = dict(
    imports=[
        'mmdet.datasets.transforms.gravel_loading',
        'mmdet.models.necks.dt_cspnext_pafpn',
        'mmdet.models.detectors.rtmdet_aux'
    ],
    allow_failed_imports=False)

# ---------------------------------------------------------------------------
# Stage-2 pipeline: no Mosaic/MixUp, with SAM2 boundary supervision
# gt_seg_map (uint8 0-255) 会被 PackDetInputs 打包为 gt_sem_seg
# ---------------------------------------------------------------------------
train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True,
         poly2mask=False),
    dict(
        type='LoadSAM2BoundarySupervision',
        sam2_root='data/gravel_big',
        split='train'),
    dict(
        type='RandomResize',
        scale=(640, 640),
        ratio_range=(0.5, 2.0),
        resize_type='Resize',
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=(640, 640),
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='Pad',
        size=(640, 640),
        pad_val=dict(img=(114, 114, 114), seg=0)),
    dict(type='PackDetInputs')
]

# ---------------------------------------------------------------------------
# Model: 继承 DT-FPN v3 neck, 添加 UAGD 边界头
# 注意: dt_loss_weight=0.0 避免 gt_sem_seg 同时被 DT loss 解读为 DT map
# DT 门控权重照常工作（前向仍然用 dt_map 做 gate），只是不再反向传播 DT 损失
# ---------------------------------------------------------------------------
model = dict(
    neck=dict(
        dt_loss_weight=0.0,
        uagd_boundary_weight=1.0,
        uagd_boundary_channels=32))

# ---------------------------------------------------------------------------
# custom_hooks: 恢复 PipelineSwitchHook（DT-FPN v3 中删掉了）
# ---------------------------------------------------------------------------
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=100,          # max_epochs - stage2_num_epochs = 120 - 20
        switch_pipeline=train_pipeline_stage2)
]

# ---------------------------------------------------------------------------
# 从已收敛的 DT-FPN v3 checkpoint 热启动
# 训练时将此路径替换为实际的最佳 checkpoint 路径
# ---------------------------------------------------------------------------
# load_from = 'work_dirs_gravel_big/rtmdet_ins_tiny_shared_dt_fpn_v3_120e_gravel_big/best_coco_segm_mAP_epoch_*.pth'

work_dir = 'work_dirs_gravel_big/rtmdet_ins_tiny_dt_fpn_uagd_v1_120e_gravel_big'

# Gravel VMamba Serial 36e Results

该文件由串行训练脚本自动维护。

## Official VMamba 2292

- config: configs/gravel/mask_rcnn_vmamba_official_2292_fpn_36e_gravel_414.py
- work_dir: work_dirs_gravel/mask_rcnn_vmamba_official_2292_fpn_36e_gravel_414
- best_ckpt: /home/yxy18034962/projects/mmdetection/work_dirs_gravel/mask_rcnn_vmamba_official_2292_fpn_36e_gravel_414/best_coco_segm_mAP_epoch_28.pth
- eval_log: [2026-04-14 23:02:24] test eval: configs/gravel/mask_rcnn_vmamba_official_2292_fpn_36e_gravel_414.py with /home/yxy18034962/projects/mmdetection/work_dirs_gravel/mask_rcnn_vmamba_official_2292_fpn_36e_gravel_414/best_coco_segm_mAP_epoch_28.pth
/home/yxy18034962/miniconda3/envs/mmdec/lib/python3.10/site-packages/timm/models/layers/__init__.py:49: FutureWarning: Importing from timm.models.layers is deprecated, please import via timm.layers
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.layers", FutureWarning)
/home/yxy18034962/projects/mmdetection/mmdet/models/backbones/vmamba_official/csms6s.py:13: UserWarning: Can not import selective_scan_cuda_oflex. This affects speed.
  warnings.warn("Can not import selective_scan_cuda_oflex. This affects speed.")
Can not import selective_scan_cuda_oflex. This affects speed.
/home/yxy18034962/projects/mmdetection/mmdet/models/backbones/vmamba_official/csms6s.py:74: FutureWarning: `torch.cuda.amp.custom_fwd(args...)` is deprecated. Please use `torch.amp.custom_fwd(args..., device_type='cuda')` instead.
  def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, oflex=True, backend=None):
/home/yxy18034962/projects/mmdetection/mmdet/models/backbones/vmamba_official/csms6s.py:91: FutureWarning: `torch.cuda.amp.custom_bwd(args...)` is deprecated. Please use `torch.amp.custom_bwd(args..., device_type='cuda')` instead.
  def backward(ctx, dout, *args):
/home/yxy18034962/projects/mmdetection/mmdet/models/backbones/vmamba_official/mamba2/ssd_combined.py:764: FutureWarning: `torch.cuda.amp.custom_fwd(args...)` is deprecated. Please use `torch.amp.custom_fwd(args..., device_type='cuda')` instead.
  def forward(ctx, zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, chunk_size, initial_states=None, seq_idx=None, dt_limit=(0.0, float("inf")), return_final_states=False, activation="silu",
/home/yxy18034962/projects/mmdetection/mmdet/models/backbones/vmamba_official/mamba2/ssd_combined.py:842: FutureWarning: `torch.cuda.amp.custom_bwd(args...)` is deprecated. Please use `torch.amp.custom_bwd(args..., device_type='cuda')` instead.
  def backward(ctx, dout, *args):
04/14 23:02:30 - mmengine - INFO - 
------------------------------------------------------------
System environment:
    sys.platform: linux
    Python: 3.10.19 (main, Oct 21 2025, 16:43:05) [GCC 11.2.0]
    CUDA available: True
    MUSA available: False
    numpy_random_seed: 875996231
    GPU 0: NVIDIA GeForce RTX 5090
    CUDA_HOME: /home/yxy18034962/miniconda3/envs/mmdec
    NVCC: Cuda compilation tools, release 12.8, V12.8.93
    GCC: gcc (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0
    PyTorch: 2.11.0.dev20251215+cu128
    PyTorch compiling details: PyTorch built with:
  - GCC 13.3
  - C++ Version: 201703
  - Intel(R) oneAPI Math Kernel Library Version 2024.2-Product Build 20240605 for Intel(R) 64 architecture applications
  - Intel(R) MKL-DNN v3.7.1 (Git Hash 8d263e693366ef8db40acc569cc7d8edf644556d)
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - LAPACK is enabled (usually provided by MKL)
  - NNPACK is enabled
  - CPU capability usage: AVX2
  - CUDA Runtime 12.8
  - NVCC architecture flags: -gencode;arch=compute_70,code=sm_70;-gencode;arch=compute_75,code=sm_75;-gencode;arch=compute_80,code=sm_80;-gencode;arch=compute_86,code=sm_86;-gencode;arch=compute_90,code=sm_90;-gencode;arch=compute_100,code=sm_100;-gencode;arch=compute_120,code=sm_120
  - CuDNN 91.0.2  (built against CUDA 12.9)
  - Magma 2.6.1
  - Build settings: BLAS_INFO=mkl, BUILD_TYPE=Release, COMMIT_SHA=ef6b76e10361515ed8641c4647fdbc93ab86ce4e, CUDA_VERSION=12.8, CUDNN_VERSION=9.10.2, CXX_COMPILER=/opt/rh/gcc-toolset-13/root/usr/bin/c++, CXX_FLAGS= -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DLIBKINETO_NOROCTRACER -DLIBKINETO_NOXPUPTI=ON -DUSE_FBGEMM -DUSE_FBGEMM_GENAI -DUSE_PYTORCH_QNNPACK -DUSE_XNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -DC10_NODEPRECATED -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=range-loop-construct -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-unknown-pragmas -Wno-unused-parameter -Wno-strict-overflow -Wno-strict-aliasing -Wno-stringop-overflow -Wsuggest-override -Wno-psabi -Wno-error=old-style-cast -faligned-new -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Wno-dangling-reference -Wno-error=dangling-reference -Wno-stringop-overflow, LAPACK_INFO=mkl, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, TORCH_VERSION=2.11.0, USE_CUDA=ON, USE_CUDNN=ON, USE_CUSPARSELT=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_GLOO=ON, USE_MKL=ON, USE_MKLDNN=ON, USE_MPI=OFF, USE_NCCL=1, USE_NNPACK=ON, USE_OPENMP=ON, USE_ROCM=OFF, USE_ROCM_KERNEL_ASSERT=OFF, USE_XCCL=OFF, USE_XPU=OFF, 

    TorchVision: 0.25.0.dev20251215+cu128
    OpenCV: 4.10.0
    MMEngine: 0.10.7

Runtime environment:
    cudnn_benchmark: False
    mp_cfg: {'mp_start_method': 'fork', 'opencv_num_threads': 0}
    dist_cfg: {'backend': 'nccl'}
    seed: 875996231
    Distributed launcher: none
    Distributed training: False
    GPU number: 1
------------------------------------------------------------

04/14 23:02:30 - mmengine - INFO - Config:
auto_scale_lr = dict(base_batch_size=16, enable=False)
backend_args = None
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'mmdet.models.backbones.vmamba_official.mmdet_vssm',
    ])
data_root = 'data/gravel_roboflow_414_mmdet/'
dataset_type = 'GravelDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=3,
        rule='greater',
        save_best='coco/segm_mAP',
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = '/home/yxy18034962/projects/mmdetection/work_dirs_gravel/mask_rcnn_vmamba_official_2292_fpn_36e_gravel_414/best_coco_segm_mAP_epoch_28.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
metainfo = dict(
    classes=('gravel', ), palette=[
        (
            255,
            140,
            0,
        ),
    ])
model = dict(
    backbone=dict(
        depths=[
            2,
            2,
            9,
            2,
        ],
        dims=[
            96,
            192,
            384,
            768,
        ],
        downsample_version='v3',
        drop_path_rate=0.2,
        forward_type='v05_noz',
        gmlp=False,
        mlp_act_layer='gelu',
        mlp_drop_rate=0.0,
        mlp_ratio=4.0,
        norm_layer='ln',
        patch_norm=True,
        patchembed_version='v2',
        pretrained='checkpoints/vmamba/vssm_tiny_0230_ckpt_epoch_262.pth',
        ssm_act_layer='silu',
        ssm_conv=3,
        ssm_conv_bias=False,
        ssm_d_state=1,
        ssm_drop_rate=0.0,
        ssm_dt_rank='auto',
        ssm_init='v0',
        ssm_ratio=2.0,
        type='MM_VMamba'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_mask=True,
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        in_channels=[
            96,
            192,
            384,
            768,
        ],
        num_outs=5,
        out_channels=256,
        type='FPN'),
    roi_head=dict(
        bbox_head=dict(
            bbox_coder=dict(
                target_means=[
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                target_stds=[
                    0.1,
                    0.1,
                    0.2,
                    0.2,
                ],
                type='DeltaXYWHBBoxCoder'),
            fc_out_channels=1024,
            in_channels=256,
            loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
            loss_cls=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
            num_classes=1,
            reg_class_agnostic=False,
            roi_feat_size=7,
            type='Shared2FCBBoxHead'),
        bbox_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(output_size=7, sampling_ratio=0, type='RoIAlign'),
            type='SingleRoIExtractor'),
        mask_head=dict(
            conv_out_channels=256,
            in_channels=256,
            loss_mask=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_mask=True),
            num_classes=1,
            num_convs=4,
            type='FCNMaskHead'),
        mask_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(output_size=14, sampling_ratio=0, type='RoIAlign'),
            type='SingleRoIExtractor'),
        type='StandardRoIHead'),
    rpn_head=dict(
        anchor_generator=dict(
            ratios=[
                0.5,
                1.0,
                2.0,
            ],
            scales=[
                8,
            ],
            strides=[
                4,
                8,
                16,
                32,
                64,
            ],
            type='AnchorGenerator'),
        bbox_coder=dict(
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            type='DeltaXYWHBBoxCoder'),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
        loss_cls=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
        type='RPNHead'),
    test_cfg=dict(
        rcnn=dict(
            mask_thr_binary=0.5,
            max_per_img=100,
            nms=dict(iou_threshold=0.5, type='nms'),
            score_thr=0.05),
        rpn=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=1000)),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=True,
                min_pos_iou=0.5,
                neg_iou_thr=0.5,
                pos_iou_thr=0.5,
                type='MaxIoUAssigner'),
            debug=False,
            mask_size=28,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=True,
                neg_pos_ub=-1,
                num=512,
                pos_fraction=0.25,
                type='RandomSampler')),
        rpn=dict(
            allowed_border=-1,
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=True,
                min_pos_iou=0.3,
                neg_iou_thr=0.3,
                pos_iou_thr=0.7,
                type='MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=False,
                neg_pos_ub=-1,
                num=256,
                pos_fraction=0.5,
                type='RandomSampler')),
        rpn_proposal=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=2000)),
    type='MaskRCNN')
optim_wrapper = dict(
    optimizer=dict(lr=0.0001, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0))),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=100, start_factor=0.001, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=36,
        gamma=0.1,
        milestones=[
            24,
            33,
        ],
        type='MultiStepLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/instances_test.json',
        backend_args=None,
        data_prefix=dict(img='test/'),
        data_root='data/gravel_roboflow_414_mmdet/',
        metainfo=dict(classes=('gravel', ), palette=[
            (
                255,
                140,
                0,
            ),
        ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                640,
                640,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='GravelDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=False,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='data/gravel_roboflow_414_mmdet/annotations/instances_test.json',
    backend_args=None,
    format_only=False,
    metric='segm',
    type='FasterCocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        640,
        640,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(max_epochs=36, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=2,
    dataset=dict(
        ann_file='annotations/instances_train.json',
        backend_args=None,
        data_prefix=dict(img='train/'),
        data_root='data/gravel_roboflow_414_mmdet/',
        filter_cfg=dict(filter_empty_gt=False, min_size=1),
        metainfo=dict(classes=('gravel', ), palette=[
            (
                255,
                140,
                0,
            ),
        ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                keep_ratio=True,
                scales=[
                    (
                        512,
                        512,
                    ),
                    (
                        640,
                        640,
                    ),
                    (
                        768,
                        768,
                    ),
                ],
                type='RandomChoiceResize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackDetInputs'),
        ],
        type='GravelDataset'),
    num_workers=2,
    persistent_workers=False,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        keep_ratio=True,
        scales=[
            (
                512,
                512,
            ),
            (
                640,
                640,
            ),
            (
                768,
                768,
            ),
        ],
        type='RandomChoiceResize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/instances_val.json',
        backend_args=None,
        data_prefix=dict(img='val/'),
        data_root='data/gravel_roboflow_414_mmdet/',
        metainfo=dict(classes=('gravel', ), palette=[
            (
                255,
                140,
                0,
            ),
        ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                640,
                640,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='GravelDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=False,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='data/gravel_roboflow_414_mmdet/annotations/instances_val.json',
    backend_args=None,
    format_only=False,
    metric='segm',
    type='FasterCocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = '/home/yxy18034962/projects/mmdetection/work_dirs_gravel/mask_rcnn_vmamba_official_2292_fpn_36e_gravel_414/test_eval'

Successfully load ckpt checkpoints/vmamba/vssm_tiny_0230_ckpt_epoch_262.pth
_IncompatibleKeys(missing_keys=['layers.2.blocks.5.norm.weight', 'layers.2.blocks.5.norm.bias', 'layers.2.blocks.5.op.x_proj_weight', 'layers.2.blocks.5.op.A_logs', 'layers.2.blocks.5.op.Ds', 'layers.2.blocks.5.op.dt_projs_weight', 'layers.2.blocks.5.op.dt_projs_bias', 'layers.2.blocks.5.op.out_norm.weight', 'layers.2.blocks.5.op.out_norm.bias', 'layers.2.blocks.5.op.in_proj.weight', 'layers.2.blocks.5.op.conv2d.weight', 'layers.2.blocks.5.op.out_proj.weight', 'layers.2.blocks.5.norm2.weight', 'layers.2.blocks.5.norm2.bias', 'layers.2.blocks.5.mlp.fc1.weight', 'layers.2.blocks.5.mlp.fc1.bias', 'layers.2.blocks.5.mlp.fc2.weight', 'layers.2.blocks.5.mlp.fc2.bias', 'layers.2.blocks.6.norm.weight', 'layers.2.blocks.6.norm.bias', 'layers.2.blocks.6.op.x_proj_weight', 'layers.2.blocks.6.op.A_logs', 'layers.2.blocks.6.op.Ds', 'layers.2.blocks.6.op.dt_projs_weight', 'layers.2.blocks.6.op.dt_projs_bias', 'layers.2.blocks.6.op.out_norm.weight', 'layers.2.blocks.6.op.out_norm.bias', 'layers.2.blocks.6.op.in_proj.weight', 'layers.2.blocks.6.op.conv2d.weight', 'layers.2.blocks.6.op.out_proj.weight', 'layers.2.blocks.6.norm2.weight', 'layers.2.blocks.6.norm2.bias', 'layers.2.blocks.6.mlp.fc1.weight', 'layers.2.blocks.6.mlp.fc1.bias', 'layers.2.blocks.6.mlp.fc2.weight', 'layers.2.blocks.6.mlp.fc2.bias', 'layers.2.blocks.7.norm.weight', 'layers.2.blocks.7.norm.bias', 'layers.2.blocks.7.op.x_proj_weight', 'layers.2.blocks.7.op.A_logs', 'layers.2.blocks.7.op.Ds', 'layers.2.blocks.7.op.dt_projs_weight', 'layers.2.blocks.7.op.dt_projs_bias', 'layers.2.blocks.7.op.out_norm.weight', 'layers.2.blocks.7.op.out_norm.bias', 'layers.2.blocks.7.op.in_proj.weight', 'layers.2.blocks.7.op.conv2d.weight', 'layers.2.blocks.7.op.out_proj.weight', 'layers.2.blocks.7.norm2.weight', 'layers.2.blocks.7.norm2.bias', 'layers.2.blocks.7.mlp.fc1.weight', 'layers.2.blocks.7.mlp.fc1.bias', 'layers.2.blocks.7.mlp.fc2.weight', 'layers.2.blocks.7.mlp.fc2.bias', 'layers.2.blocks.8.norm.weight', 'layers.2.blocks.8.norm.bias', 'layers.2.blocks.8.op.x_proj_weight', 'layers.2.blocks.8.op.A_logs', 'layers.2.blocks.8.op.Ds', 'layers.2.blocks.8.op.dt_projs_weight', 'layers.2.blocks.8.op.dt_projs_bias', 'layers.2.blocks.8.op.out_norm.weight', 'layers.2.blocks.8.op.out_norm.bias', 'layers.2.blocks.8.op.in_proj.weight', 'layers.2.blocks.8.op.conv2d.weight', 'layers.2.blocks.8.op.out_proj.weight', 'layers.2.blocks.8.norm2.weight', 'layers.2.blocks.8.norm2.bias', 'layers.2.blocks.8.mlp.fc1.weight', 'layers.2.blocks.8.mlp.fc1.bias', 'layers.2.blocks.8.mlp.fc2.weight', 'layers.2.blocks.8.mlp.fc2.bias', 'outnorm0.weight', 'outnorm0.bias', 'outnorm1.weight', 'outnorm1.bias', 'outnorm2.weight', 'outnorm2.bias', 'outnorm3.weight', 'outnorm3.bias'], unexpected_keys=['classifier.norm.weight', 'classifier.norm.bias', 'classifier.head.weight', 'classifier.head.bias'])
04/14 23:02:31 - mmengine - INFO - Distributed training is not used, all SyncBatchNorm (SyncBN) layers in the model will be automatically reverted to BatchNormXd layers if they are used.
04/14 23:02:31 - mmengine - INFO - Hooks will be executed in the following order:
before_run:
(VERY_HIGH   ) RuntimeInfoHook                    
(BELOW_NORMAL) LoggerHook                         
 -------------------- 
before_train:
(VERY_HIGH   ) RuntimeInfoHook                    
(NORMAL      ) IterTimerHook                      
(VERY_LOW    ) CheckpointHook                     
 -------------------- 
before_train_epoch:
(VERY_HIGH   ) RuntimeInfoHook                    
(NORMAL      ) IterTimerHook                      
(NORMAL      ) DistSamplerSeedHook                
 -------------------- 
before_train_iter:
(VERY_HIGH   ) RuntimeInfoHook                    
(NORMAL      ) IterTimerHook                      
 -------------------- 
after_train_iter:
(VERY_HIGH   ) RuntimeInfoHook                    
(NORMAL      ) IterTimerHook                      
(BELOW_NORMAL) LoggerHook                         
(LOW         ) ParamSchedulerHook                 
(VERY_LOW    ) CheckpointHook                     
 -------------------- 
after_train_epoch:
(NORMAL      ) IterTimerHook                      
(LOW         ) ParamSchedulerHook                 
(VERY_LOW    ) CheckpointHook                     
 -------------------- 
before_val:
(VERY_HIGH   ) RuntimeInfoHook                    
 -------------------- 
before_val_epoch:
(NORMAL      ) IterTimerHook                      
 -------------------- 
before_val_iter:
(NORMAL      ) IterTimerHook                      
 -------------------- 
after_val_iter:
(NORMAL      ) IterTimerHook                      
(NORMAL      ) DetVisualizationHook               
(BELOW_NORMAL) LoggerHook                         
 -------------------- 
after_val_epoch:
(VERY_HIGH   ) RuntimeInfoHook                    
(NORMAL      ) IterTimerHook                      
(BELOW_NORMAL) LoggerHook                         
(LOW         ) ParamSchedulerHook                 
(VERY_LOW    ) CheckpointHook                     
 -------------------- 
after_val:
(VERY_HIGH   ) RuntimeInfoHook                    
 -------------------- 
after_train:
(VERY_HIGH   ) RuntimeInfoHook                    
(VERY_LOW    ) CheckpointHook                     
 -------------------- 
before_test:
(VERY_HIGH   ) RuntimeInfoHook                    
 -------------------- 
before_test_epoch:
(NORMAL      ) IterTimerHook                      
 -------------------- 
before_test_iter:
(NORMAL      ) IterTimerHook                      
 -------------------- 
after_test_iter:
(NORMAL      ) IterTimerHook                      
(NORMAL      ) DetVisualizationHook               
(BELOW_NORMAL) LoggerHook                         
 -------------------- 
after_test_epoch:
(VERY_HIGH   ) RuntimeInfoHook                    
(NORMAL      ) IterTimerHook                      
(BELOW_NORMAL) LoggerHook                         
 -------------------- 
after_test:
(VERY_HIGH   ) RuntimeInfoHook                    
 -------------------- 
after_run:
(BELOW_NORMAL) LoggerHook                         
 -------------------- 
loading annotations into memory...
Done (t=0.02s)
creating index...
index created!
loading annotations into memory...
Done (t=0.02s)
creating index...
index created!
Loads checkpoint by local backend from path: /home/yxy18034962/projects/mmdetection/work_dirs_gravel/mask_rcnn_vmamba_official_2292_fpn_36e_gravel_414/best_coco_segm_mAP_epoch_28.pth
04/14 23:02:32 - mmengine - INFO - Load checkpoint from /home/yxy18034962/projects/mmdetection/work_dirs_gravel/mask_rcnn_vmamba_official_2292_fpn_36e_gravel_414/best_coco_segm_mAP_epoch_28.pth
04/14 23:02:38 - mmengine - INFO - Epoch(test) [50/50]    eta: 0:00:00  time: 0.1341  data_time: 0.0790  memory: 790  
04/14 23:02:39 - mmengine - INFO - Evaluating segm...
04/14 23:02:39 - mmengine - INFO - Evaluate annotation type *segm*
04/14 23:02:39 - mmengine - INFO - COCOeval_opt.evaluate() finished...
04/14 23:02:39 - mmengine - INFO - DONE (t=0.12s).
04/14 23:02:39 - mmengine - INFO - Accumulating evaluation results...
04/14 23:02:39 - mmengine - INFO - COCOeval_opt.accumulate() finished...
04/14 23:02:39 - mmengine - INFO - DONE (t=0.00s).
04/14 23:02:39 - mmengine - INFO -  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.275
04/14 23:02:39 - mmengine - INFO -  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.397
04/14 23:02:39 - mmengine - INFO -  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.324
04/14 23:02:39 - mmengine - INFO -  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.241
04/14 23:02:39 - mmengine - INFO -  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.591
04/14 23:02:39 - mmengine - INFO -  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.702
04/14 23:02:39 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.296
04/14 23:02:39 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.296
04/14 23:02:39 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.296
04/14 23:02:39 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.262
04/14 23:02:39 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.625
04/14 23:02:39 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.719
04/14 23:02:39 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.401
04/14 23:02:39 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.347
04/14 23:02:39 - mmengine - INFO - segm_mAP_copypaste: 0.275 0.397 0.324 0.241 0.591 0.702
04/14 23:02:39 - mmengine - INFO - Epoch(test) [50/50]    coco/segm_mAP: 0.2750  coco/segm_mAP_50: 0.3970  coco/segm_mAP_75: 0.3240  coco/segm_mAP_s: 0.2410  coco/segm_mAP_m: 0.5910  coco/segm_mAP_l: 0.7020  data_time: 0.0790  time: 0.1341
/home/yxy18034962/projects/mmdetection/work_dirs_gravel/mask_rcnn_vmamba_official_2292_fpn_36e_gravel_414/test_eval.log

## RS-LightMamba S4 GlobalAttn

- config: configs/gravel/mask_rcnn_rs_lightmamba_s4_global_attn_fpn_36e_gravel_414.py
- work_dir: work_dirs_gravel/mask_rcnn_rs_lightmamba_s4_global_attn_fpn_36e_gravel_414
- best_ckpt: /home/yxy18034962/projects/mmdetection/work_dirs_gravel/mask_rcnn_rs_lightmamba_s4_global_attn_fpn_36e_gravel_414/best_coco_segm_mAP_epoch_27.pth
- eval_log: [2026-04-15 00:55:08] test eval: configs/gravel/mask_rcnn_rs_lightmamba_s4_global_attn_fpn_36e_gravel_414.py with /home/yxy18034962/projects/mmdetection/work_dirs_gravel/mask_rcnn_rs_lightmamba_s4_global_attn_fpn_36e_gravel_414/best_coco_segm_mAP_epoch_27.pth
/home/yxy18034962/miniconda3/envs/mmdec/lib/python3.10/site-packages/timm/models/layers/__init__.py:49: FutureWarning: Importing from timm.models.layers is deprecated, please import via timm.layers
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.layers", FutureWarning)
/home/yxy18034962/projects/mmdetection/mmdet/models/backbones/vmamba_official/csms6s.py:13: UserWarning: Can not import selective_scan_cuda_oflex. This affects speed.
  warnings.warn("Can not import selective_scan_cuda_oflex. This affects speed.")
Can not import selective_scan_cuda_oflex. This affects speed.
/home/yxy18034962/projects/mmdetection/mmdet/models/backbones/vmamba_official/csms6s.py:74: FutureWarning: `torch.cuda.amp.custom_fwd(args...)` is deprecated. Please use `torch.amp.custom_fwd(args..., device_type='cuda')` instead.
  def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, oflex=True, backend=None):
/home/yxy18034962/projects/mmdetection/mmdet/models/backbones/vmamba_official/csms6s.py:91: FutureWarning: `torch.cuda.amp.custom_bwd(args...)` is deprecated. Please use `torch.amp.custom_bwd(args..., device_type='cuda')` instead.
  def backward(ctx, dout, *args):
/home/yxy18034962/projects/mmdetection/mmdet/models/backbones/vmamba_official/mamba2/ssd_combined.py:764: FutureWarning: `torch.cuda.amp.custom_fwd(args...)` is deprecated. Please use `torch.amp.custom_fwd(args..., device_type='cuda')` instead.
  def forward(ctx, zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, chunk_size, initial_states=None, seq_idx=None, dt_limit=(0.0, float("inf")), return_final_states=False, activation="silu",
/home/yxy18034962/projects/mmdetection/mmdet/models/backbones/vmamba_official/mamba2/ssd_combined.py:842: FutureWarning: `torch.cuda.amp.custom_bwd(args...)` is deprecated. Please use `torch.amp.custom_bwd(args..., device_type='cuda')` instead.
  def backward(ctx, dout, *args):
04/15 00:55:11 - mmengine - INFO - 
------------------------------------------------------------
System environment:
    sys.platform: linux
    Python: 3.10.19 (main, Oct 21 2025, 16:43:05) [GCC 11.2.0]
    CUDA available: True
    MUSA available: False
    numpy_random_seed: 646658287
    GPU 0: NVIDIA GeForce RTX 5090
    CUDA_HOME: /home/yxy18034962/miniconda3/envs/mmdec
    NVCC: Cuda compilation tools, release 12.8, V12.8.93
    GCC: gcc (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0
    PyTorch: 2.11.0.dev20251215+cu128
    PyTorch compiling details: PyTorch built with:
  - GCC 13.3
  - C++ Version: 201703
  - Intel(R) oneAPI Math Kernel Library Version 2024.2-Product Build 20240605 for Intel(R) 64 architecture applications
  - Intel(R) MKL-DNN v3.7.1 (Git Hash 8d263e693366ef8db40acc569cc7d8edf644556d)
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - LAPACK is enabled (usually provided by MKL)
  - NNPACK is enabled
  - CPU capability usage: AVX2
  - CUDA Runtime 12.8
  - NVCC architecture flags: -gencode;arch=compute_70,code=sm_70;-gencode;arch=compute_75,code=sm_75;-gencode;arch=compute_80,code=sm_80;-gencode;arch=compute_86,code=sm_86;-gencode;arch=compute_90,code=sm_90;-gencode;arch=compute_100,code=sm_100;-gencode;arch=compute_120,code=sm_120
  - CuDNN 91.0.2  (built against CUDA 12.9)
  - Magma 2.6.1
  - Build settings: BLAS_INFO=mkl, BUILD_TYPE=Release, COMMIT_SHA=ef6b76e10361515ed8641c4647fdbc93ab86ce4e, CUDA_VERSION=12.8, CUDNN_VERSION=9.10.2, CXX_COMPILER=/opt/rh/gcc-toolset-13/root/usr/bin/c++, CXX_FLAGS= -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DLIBKINETO_NOROCTRACER -DLIBKINETO_NOXPUPTI=ON -DUSE_FBGEMM -DUSE_FBGEMM_GENAI -DUSE_PYTORCH_QNNPACK -DUSE_XNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -DC10_NODEPRECATED -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=range-loop-construct -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-unknown-pragmas -Wno-unused-parameter -Wno-strict-overflow -Wno-strict-aliasing -Wno-stringop-overflow -Wsuggest-override -Wno-psabi -Wno-error=old-style-cast -faligned-new -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Wno-dangling-reference -Wno-error=dangling-reference -Wno-stringop-overflow, LAPACK_INFO=mkl, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, TORCH_VERSION=2.11.0, USE_CUDA=ON, USE_CUDNN=ON, USE_CUSPARSELT=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_GLOO=ON, USE_MKL=ON, USE_MKLDNN=ON, USE_MPI=OFF, USE_NCCL=1, USE_NNPACK=ON, USE_OPENMP=ON, USE_ROCM=OFF, USE_ROCM_KERNEL_ASSERT=OFF, USE_XCCL=OFF, USE_XPU=OFF, 

    TorchVision: 0.25.0.dev20251215+cu128
    OpenCV: 4.10.0
    MMEngine: 0.10.7

Runtime environment:
    cudnn_benchmark: False
    mp_cfg: {'mp_start_method': 'fork', 'opencv_num_threads': 0}
    dist_cfg: {'backend': 'nccl'}
    seed: 646658287
    Distributed launcher: none
    Distributed training: False
    GPU number: 1
------------------------------------------------------------

04/15 00:55:11 - mmengine - INFO - Config:
auto_scale_lr = dict(base_batch_size=16, enable=False)
backend_args = None
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'mmdet.models.backbones.rs_lightmamba.lightmamba_backbone',
        'mmdet.models.detectors.rs_ig_mask_rcnn',
    ])
data_root = 'data/gravel_roboflow_414_mmdet/'
dataset_type = 'GravelDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=3,
        rule='greater',
        save_best='coco/segm_mAP',
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = '/home/yxy18034962/projects/mmdetection/work_dirs_gravel/mask_rcnn_rs_lightmamba_s4_global_attn_fpn_36e_gravel_414/best_coco_segm_mAP_epoch_27.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
metainfo = dict(
    classes=('gravel', ), palette=[
        (
            255,
            140,
            0,
        ),
    ])
model = dict(
    backbone=dict(
        attention_attn_drop=0.0,
        attention_fg_lk_size=7,
        attention_fg_loss_weight=0.2,
        attention_fg_norm_type='bn',
        attention_fg_stage=3,
        attention_mlp_ratio=4.0,
        attention_num_heads=8,
        attention_proj_drop=0.0,
        attention_qkv_bias=True,
        attention_stages=[
            3,
        ],
        attention_use_fg_loss=True,
        depths=[
            2,
            2,
            9,
            2,
        ],
        dims=[
            96,
            192,
            384,
            768,
        ],
        downsample_version='v3',
        drop_path_rate=0.2,
        forward_type='v05_noz',
        gmlp=False,
        mlp_act_layer='gelu',
        mlp_drop_rate=0.0,
        mlp_ratio=4.0,
        norm_layer='ln',
        official_pretrained=
        'checkpoints/vmamba/vssm_tiny_0230_ckpt_epoch_262.pth',
        patch_norm=True,
        patchembed_version='v2',
        pretrained_key='model',
        ssm_act_layer='silu',
        ssm_conv=3,
        ssm_conv_bias=False,
        ssm_d_state=1,
        ssm_drop_rate=0.0,
        ssm_dt_rank='auto',
        ssm_init='v0',
        ssm_ratio=2.0,
        strict_pretrained=False,
        type='RSLightMambaBackbone'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_mask=True,
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        in_channels=[
            96,
            192,
            384,
            768,
        ],
        num_outs=5,
        out_channels=256,
        type='FPN'),
    roi_head=dict(
        bbox_head=dict(
            bbox_coder=dict(
                target_means=[
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                target_stds=[
                    0.1,
                    0.1,
                    0.2,
                    0.2,
                ],
                type='DeltaXYWHBBoxCoder'),
            fc_out_channels=1024,
            in_channels=256,
            loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
            loss_cls=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
            num_classes=1,
            reg_class_agnostic=False,
            roi_feat_size=7,
            type='Shared2FCBBoxHead'),
        bbox_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(output_size=7, sampling_ratio=0, type='RoIAlign'),
            type='SingleRoIExtractor'),
        mask_head=dict(
            conv_out_channels=256,
            in_channels=256,
            loss_mask=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_mask=True),
            num_classes=1,
            num_convs=4,
            type='FCNMaskHead'),
        mask_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(output_size=14, sampling_ratio=0, type='RoIAlign'),
            type='SingleRoIExtractor'),
        type='StandardRoIHead'),
    rpn_head=dict(
        anchor_generator=dict(
            ratios=[
                0.5,
                1.0,
                2.0,
            ],
            scales=[
                8,
            ],
            strides=[
                4,
                8,
                16,
                32,
                64,
            ],
            type='AnchorGenerator'),
        bbox_coder=dict(
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            type='DeltaXYWHBBoxCoder'),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
        loss_cls=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
        type='RPNHead'),
    test_cfg=dict(
        rcnn=dict(
            mask_thr_binary=0.5,
            max_per_img=100,
            nms=dict(iou_threshold=0.5, type='nms'),
            score_thr=0.05),
        rpn=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=1000)),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=True,
                min_pos_iou=0.5,
                neg_iou_thr=0.5,
                pos_iou_thr=0.5,
                type='MaxIoUAssigner'),
            debug=False,
            mask_size=28,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=True,
                neg_pos_ub=-1,
                num=512,
                pos_fraction=0.25,
                type='RandomSampler')),
        rpn=dict(
            allowed_border=-1,
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=True,
                min_pos_iou=0.3,
                neg_iou_thr=0.3,
                pos_iou_thr=0.7,
                type='MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=False,
                neg_pos_ub=-1,
                num=256,
                pos_fraction=0.5,
                type='RandomSampler')),
        rpn_proposal=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=2000)),
    type='RSMaskRCNN')
optim_wrapper = dict(
    optimizer=dict(lr=0.0001, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0))),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=100, start_factor=0.001, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=36,
        gamma=0.1,
        milestones=[
            24,
            33,
        ],
        type='MultiStepLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/instances_test.json',
        backend_args=None,
        data_prefix=dict(img='test/'),
        data_root='data/gravel_roboflow_414_mmdet/',
        metainfo=dict(classes=('gravel', ), palette=[
            (
                255,
                140,
                0,
            ),
        ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                640,
                640,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='GravelDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=False,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='data/gravel_roboflow_414_mmdet/annotations/instances_test.json',
    backend_args=None,
    format_only=False,
    metric='segm',
    type='FasterCocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        640,
        640,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(max_epochs=36, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=2,
    dataset=dict(
        ann_file='annotations/instances_train.json',
        backend_args=None,
        data_prefix=dict(img='train/'),
        data_root='data/gravel_roboflow_414_mmdet/',
        filter_cfg=dict(filter_empty_gt=False, min_size=1),
        metainfo=dict(classes=('gravel', ), palette=[
            (
                255,
                140,
                0,
            ),
        ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                keep_ratio=True,
                scales=[
                    (
                        512,
                        512,
                    ),
                    (
                        640,
                        640,
                    ),
                    (
                        768,
                        768,
                    ),
                ],
                type='RandomChoiceResize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackDetInputs'),
        ],
        type='GravelDataset'),
    num_workers=2,
    persistent_workers=False,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        keep_ratio=True,
        scales=[
            (
                512,
                512,
            ),
            (
                640,
                640,
            ),
            (
                768,
                768,
            ),
        ],
        type='RandomChoiceResize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/instances_val.json',
        backend_args=None,
        data_prefix=dict(img='val/'),
        data_root='data/gravel_roboflow_414_mmdet/',
        metainfo=dict(classes=('gravel', ), palette=[
            (
                255,
                140,
                0,
            ),
        ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                640,
                640,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='GravelDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=False,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='data/gravel_roboflow_414_mmdet/annotations/instances_val.json',
    backend_args=None,
    format_only=False,
    metric='segm',
    type='FasterCocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = '/home/yxy18034962/projects/mmdetection/work_dirs_gravel/mask_rcnn_rs_lightmamba_s4_global_attn_fpn_36e_gravel_414/test_eval'

[Stage4-Attn] Replaced 2 VSS blocks in stages [3] with global attention.
Loaded official checkpoint: checkpoints/vmamba/vssm_tiny_0230_ckpt_epoch_262.pth
Loaded params: 14.494M / 36.660M (39.54%)
Exact params: 14.494M, Partial params: 0.000M, Skipped keys: 0
Missing keys: 127
Unexpected keys: 0
04/15 00:55:12 - mmengine - INFO - Distributed training is not used, all SyncBatchNorm (SyncBN) layers in the model will be automatically reverted to BatchNormXd layers if they are used.
04/15 00:55:12 - mmengine - INFO - Hooks will be executed in the following order:
before_run:
(VERY_HIGH   ) RuntimeInfoHook                    
(BELOW_NORMAL) LoggerHook                         
 -------------------- 
before_train:
(VERY_HIGH   ) RuntimeInfoHook                    
(NORMAL      ) IterTimerHook                      
(VERY_LOW    ) CheckpointHook                     
 -------------------- 
before_train_epoch:
(VERY_HIGH   ) RuntimeInfoHook                    
(NORMAL      ) IterTimerHook                      
(NORMAL      ) DistSamplerSeedHook                
 -------------------- 
before_train_iter:
(VERY_HIGH   ) RuntimeInfoHook                    
(NORMAL      ) IterTimerHook                      
 -------------------- 
after_train_iter:
(VERY_HIGH   ) RuntimeInfoHook                    
(NORMAL      ) IterTimerHook                      
(BELOW_NORMAL) LoggerHook                         
(LOW         ) ParamSchedulerHook                 
(VERY_LOW    ) CheckpointHook                     
 -------------------- 
after_train_epoch:
(NORMAL      ) IterTimerHook                      
(LOW         ) ParamSchedulerHook                 
(VERY_LOW    ) CheckpointHook                     
 -------------------- 
before_val:
(VERY_HIGH   ) RuntimeInfoHook                    
 -------------------- 
before_val_epoch:
(NORMAL      ) IterTimerHook                      
 -------------------- 
before_val_iter:
(NORMAL      ) IterTimerHook                      
 -------------------- 
after_val_iter:
(NORMAL      ) IterTimerHook                      
(NORMAL      ) DetVisualizationHook               
(BELOW_NORMAL) LoggerHook                         
 -------------------- 
after_val_epoch:
(VERY_HIGH   ) RuntimeInfoHook                    
(NORMAL      ) IterTimerHook                      
(BELOW_NORMAL) LoggerHook                         
(LOW         ) ParamSchedulerHook                 
(VERY_LOW    ) CheckpointHook                     
 -------------------- 
after_val:
(VERY_HIGH   ) RuntimeInfoHook                    
 -------------------- 
after_train:
(VERY_HIGH   ) RuntimeInfoHook                    
(VERY_LOW    ) CheckpointHook                     
 -------------------- 
before_test:
(VERY_HIGH   ) RuntimeInfoHook                    
 -------------------- 
before_test_epoch:
(NORMAL      ) IterTimerHook                      
 -------------------- 
before_test_iter:
(NORMAL      ) IterTimerHook                      
 -------------------- 
after_test_iter:
(NORMAL      ) IterTimerHook                      
(NORMAL      ) DetVisualizationHook               
(BELOW_NORMAL) LoggerHook                         
 -------------------- 
after_test_epoch:
(VERY_HIGH   ) RuntimeInfoHook                    
(NORMAL      ) IterTimerHook                      
(BELOW_NORMAL) LoggerHook                         
 -------------------- 
after_test:
(VERY_HIGH   ) RuntimeInfoHook                    
 -------------------- 
after_run:
(BELOW_NORMAL) LoggerHook                         
 -------------------- 
loading annotations into memory...
Done (t=0.02s)
creating index...
index created!
loading annotations into memory...
Done (t=0.02s)
creating index...
index created!
Loads checkpoint by local backend from path: /home/yxy18034962/projects/mmdetection/work_dirs_gravel/mask_rcnn_rs_lightmamba_s4_global_attn_fpn_36e_gravel_414/best_coco_segm_mAP_epoch_27.pth
04/15 00:55:13 - mmengine - INFO - Load checkpoint from /home/yxy18034962/projects/mmdetection/work_dirs_gravel/mask_rcnn_rs_lightmamba_s4_global_attn_fpn_36e_gravel_414/best_coco_segm_mAP_epoch_27.pth
04/15 00:55:19 - mmengine - INFO - Epoch(test) [50/50]    eta: 0:00:00  time: 0.1218  data_time: 0.0693  memory: 780  
04/15 00:55:19 - mmengine - INFO - Evaluating segm...
04/15 00:55:19 - mmengine - INFO - Evaluate annotation type *segm*
04/15 00:55:19 - mmengine - INFO - COCOeval_opt.evaluate() finished...
04/15 00:55:19 - mmengine - INFO - DONE (t=0.12s).
04/15 00:55:19 - mmengine - INFO - Accumulating evaluation results...
04/15 00:55:19 - mmengine - INFO - COCOeval_opt.accumulate() finished...
04/15 00:55:19 - mmengine - INFO - DONE (t=0.00s).
04/15 00:55:19 - mmengine - INFO -  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.275
04/15 00:55:19 - mmengine - INFO -  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.398
04/15 00:55:19 - mmengine - INFO -  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.324
04/15 00:55:19 - mmengine - INFO -  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.241
04/15 00:55:19 - mmengine - INFO -  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.591
04/15 00:55:19 - mmengine - INFO -  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.683
04/15 00:55:19 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.297
04/15 00:55:19 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.297
04/15 00:55:19 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.297
04/15 00:55:19 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.263
04/15 00:55:19 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.627
04/15 00:55:19 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.702
04/15 00:55:19 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.402
04/15 00:55:19 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.348
04/15 00:55:19 - mmengine - INFO - segm_mAP_copypaste: 0.275 0.398 0.324 0.241 0.591 0.683
04/15 00:55:19 - mmengine - INFO - Epoch(test) [50/50]    coco/segm_mAP: 0.2750  coco/segm_mAP_50: 0.3980  coco/segm_mAP_75: 0.3240  coco/segm_mAP_s: 0.2410  coco/segm_mAP_m: 0.5910  coco/segm_mAP_l: 0.6830  data_time: 0.0693  time: 0.1218
/home/yxy18034962/projects/mmdetection/work_dirs_gravel/mask_rcnn_rs_lightmamba_s4_global_attn_fpn_36e_gravel_414/test_eval.log

## RS-LightMamba S4 GlobalAttn HF-FPN

- config: configs/gravel/mask_rcnn_rs_lightmamba_s4_global_attn_hf_fpn_36e_gravel_414.py
- work_dir: work_dirs_gravel/mask_rcnn_rs_lightmamba_s4_global_attn_hf_fpn_36e_gravel_414
- best_ckpt: /home/yxy18034962/projects/mmdetection/work_dirs_gravel/mask_rcnn_rs_lightmamba_s4_global_attn_hf_fpn_36e_gravel_414/best_coco_segm_mAP_epoch_26.pth
- eval_log: [2026-04-15 02:45:48] test eval: configs/gravel/mask_rcnn_rs_lightmamba_s4_global_attn_hf_fpn_36e_gravel_414.py with /home/yxy18034962/projects/mmdetection/work_dirs_gravel/mask_rcnn_rs_lightmamba_s4_global_attn_hf_fpn_36e_gravel_414/best_coco_segm_mAP_epoch_26.pth
/home/yxy18034962/miniconda3/envs/mmdec/lib/python3.10/site-packages/timm/models/layers/__init__.py:49: FutureWarning: Importing from timm.models.layers is deprecated, please import via timm.layers
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.layers", FutureWarning)
/home/yxy18034962/projects/mmdetection/mmdet/models/backbones/vmamba_official/csms6s.py:13: UserWarning: Can not import selective_scan_cuda_oflex. This affects speed.
  warnings.warn("Can not import selective_scan_cuda_oflex. This affects speed.")
Can not import selective_scan_cuda_oflex. This affects speed.
/home/yxy18034962/projects/mmdetection/mmdet/models/backbones/vmamba_official/csms6s.py:74: FutureWarning: `torch.cuda.amp.custom_fwd(args...)` is deprecated. Please use `torch.amp.custom_fwd(args..., device_type='cuda')` instead.
  def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, oflex=True, backend=None):
/home/yxy18034962/projects/mmdetection/mmdet/models/backbones/vmamba_official/csms6s.py:91: FutureWarning: `torch.cuda.amp.custom_bwd(args...)` is deprecated. Please use `torch.amp.custom_bwd(args..., device_type='cuda')` instead.
  def backward(ctx, dout, *args):
/home/yxy18034962/projects/mmdetection/mmdet/models/backbones/vmamba_official/mamba2/ssd_combined.py:764: FutureWarning: `torch.cuda.amp.custom_fwd(args...)` is deprecated. Please use `torch.amp.custom_fwd(args..., device_type='cuda')` instead.
  def forward(ctx, zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, chunk_size, initial_states=None, seq_idx=None, dt_limit=(0.0, float("inf")), return_final_states=False, activation="silu",
/home/yxy18034962/projects/mmdetection/mmdet/models/backbones/vmamba_official/mamba2/ssd_combined.py:842: FutureWarning: `torch.cuda.amp.custom_bwd(args...)` is deprecated. Please use `torch.amp.custom_bwd(args..., device_type='cuda')` instead.
  def backward(ctx, dout, *args):
04/15 02:45:51 - mmengine - INFO - 
------------------------------------------------------------
System environment:
    sys.platform: linux
    Python: 3.10.19 (main, Oct 21 2025, 16:43:05) [GCC 11.2.0]
    CUDA available: True
    MUSA available: False
    numpy_random_seed: 964516539
    GPU 0: NVIDIA GeForce RTX 5090
    CUDA_HOME: /home/yxy18034962/miniconda3/envs/mmdec
    NVCC: Cuda compilation tools, release 12.8, V12.8.93
    GCC: gcc (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0
    PyTorch: 2.11.0.dev20251215+cu128
    PyTorch compiling details: PyTorch built with:
  - GCC 13.3
  - C++ Version: 201703
  - Intel(R) oneAPI Math Kernel Library Version 2024.2-Product Build 20240605 for Intel(R) 64 architecture applications
  - Intel(R) MKL-DNN v3.7.1 (Git Hash 8d263e693366ef8db40acc569cc7d8edf644556d)
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - LAPACK is enabled (usually provided by MKL)
  - NNPACK is enabled
  - CPU capability usage: AVX2
  - CUDA Runtime 12.8
  - NVCC architecture flags: -gencode;arch=compute_70,code=sm_70;-gencode;arch=compute_75,code=sm_75;-gencode;arch=compute_80,code=sm_80;-gencode;arch=compute_86,code=sm_86;-gencode;arch=compute_90,code=sm_90;-gencode;arch=compute_100,code=sm_100;-gencode;arch=compute_120,code=sm_120
  - CuDNN 91.0.2  (built against CUDA 12.9)
  - Magma 2.6.1
  - Build settings: BLAS_INFO=mkl, BUILD_TYPE=Release, COMMIT_SHA=ef6b76e10361515ed8641c4647fdbc93ab86ce4e, CUDA_VERSION=12.8, CUDNN_VERSION=9.10.2, CXX_COMPILER=/opt/rh/gcc-toolset-13/root/usr/bin/c++, CXX_FLAGS= -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DLIBKINETO_NOROCTRACER -DLIBKINETO_NOXPUPTI=ON -DUSE_FBGEMM -DUSE_FBGEMM_GENAI -DUSE_PYTORCH_QNNPACK -DUSE_XNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -DC10_NODEPRECATED -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=range-loop-construct -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-unknown-pragmas -Wno-unused-parameter -Wno-strict-overflow -Wno-strict-aliasing -Wno-stringop-overflow -Wsuggest-override -Wno-psabi -Wno-error=old-style-cast -faligned-new -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Wno-dangling-reference -Wno-error=dangling-reference -Wno-stringop-overflow, LAPACK_INFO=mkl, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, TORCH_VERSION=2.11.0, USE_CUDA=ON, USE_CUDNN=ON, USE_CUSPARSELT=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_GLOO=ON, USE_MKL=ON, USE_MKLDNN=ON, USE_MPI=OFF, USE_NCCL=1, USE_NNPACK=ON, USE_OPENMP=ON, USE_ROCM=OFF, USE_ROCM_KERNEL_ASSERT=OFF, USE_XCCL=OFF, USE_XPU=OFF, 

    TorchVision: 0.25.0.dev20251215+cu128
    OpenCV: 4.10.0
    MMEngine: 0.10.7

Runtime environment:
    cudnn_benchmark: False
    mp_cfg: {'mp_start_method': 'fork', 'opencv_num_threads': 0}
    dist_cfg: {'backend': 'nccl'}
    seed: 964516539
    Distributed launcher: none
    Distributed training: False
    GPU number: 1
------------------------------------------------------------

04/15 02:45:51 - mmengine - INFO - Config:
auto_scale_lr = dict(base_batch_size=16, enable=False)
backend_args = None
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'mmdet.models.backbones.rs_lightmamba.lightmamba_backbone',
        'mmdet.models.detectors.rs_ig_mask_rcnn',
    ])
data_root = 'data/gravel_roboflow_414_mmdet/'
dataset_type = 'GravelDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=3,
        rule='greater',
        save_best='coco/segm_mAP',
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = '/home/yxy18034962/projects/mmdetection/work_dirs_gravel/mask_rcnn_rs_lightmamba_s4_global_attn_hf_fpn_36e_gravel_414/best_coco_segm_mAP_epoch_26.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
metainfo = dict(
    classes=('gravel', ), palette=[
        (
            255,
            140,
            0,
        ),
    ])
model = dict(
    backbone=dict(
        attention_attn_drop=0.0,
        attention_fg_lk_size=7,
        attention_fg_loss_weight=0.2,
        attention_fg_norm_type='bn',
        attention_fg_stage=3,
        attention_mlp_ratio=4.0,
        attention_num_heads=8,
        attention_proj_drop=0.0,
        attention_qkv_bias=True,
        attention_stages=[
            3,
        ],
        attention_use_fg_loss=True,
        depths=[
            2,
            2,
            9,
            2,
        ],
        dims=[
            96,
            192,
            384,
            768,
        ],
        downsample_version='v3',
        drop_path_rate=0.2,
        forward_type='v05_noz',
        gmlp=False,
        hf_map_stages=[
            0,
            1,
            2,
        ],
        mlp_act_layer='gelu',
        mlp_drop_rate=0.0,
        mlp_ratio=4.0,
        norm_layer='ln',
        official_pretrained=
        'checkpoints/vmamba/vssm_tiny_0230_ckpt_epoch_262.pth',
        output_hf_maps=True,
        patch_norm=True,
        patchembed_version='v2',
        pretrained_key='model',
        ssm_act_layer='silu',
        ssm_conv=3,
        ssm_conv_bias=False,
        ssm_d_state=1,
        ssm_drop_rate=0.0,
        ssm_dt_rank='auto',
        ssm_init='v0',
        ssm_ratio=2.0,
        strict_pretrained=False,
        type='RSLightMambaBackbone'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_mask=True,
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        in_channels=[
            96,
            192,
            384,
            768,
        ],
        num_outs=5,
        out_channels=256,
        type='HF_FPN'),
    roi_head=dict(
        bbox_head=dict(
            bbox_coder=dict(
                target_means=[
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                target_stds=[
                    0.1,
                    0.1,
                    0.2,
                    0.2,
                ],
                type='DeltaXYWHBBoxCoder'),
            fc_out_channels=1024,
            in_channels=256,
            loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
            loss_cls=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
            num_classes=1,
            reg_class_agnostic=False,
            roi_feat_size=7,
            type='Shared2FCBBoxHead'),
        bbox_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(output_size=7, sampling_ratio=0, type='RoIAlign'),
            type='SingleRoIExtractor'),
        mask_head=dict(
            conv_out_channels=256,
            in_channels=256,
            loss_mask=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_mask=True),
            num_classes=1,
            num_convs=4,
            type='FCNMaskHead'),
        mask_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(output_size=14, sampling_ratio=0, type='RoIAlign'),
            type='SingleRoIExtractor'),
        type='StandardRoIHead'),
    rpn_head=dict(
        anchor_generator=dict(
            ratios=[
                0.5,
                1.0,
                2.0,
            ],
            scales=[
                8,
            ],
            strides=[
                4,
                8,
                16,
                32,
                64,
            ],
            type='AnchorGenerator'),
        bbox_coder=dict(
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            type='DeltaXYWHBBoxCoder'),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
        loss_cls=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
        type='RPNHead'),
    test_cfg=dict(
        rcnn=dict(
            mask_thr_binary=0.5,
            max_per_img=100,
            nms=dict(iou_threshold=0.5, type='nms'),
            score_thr=0.05),
        rpn=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=1000)),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=True,
                min_pos_iou=0.5,
                neg_iou_thr=0.5,
                pos_iou_thr=0.5,
                type='MaxIoUAssigner'),
            debug=False,
            mask_size=28,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=True,
                neg_pos_ub=-1,
                num=512,
                pos_fraction=0.25,
                type='RandomSampler')),
        rpn=dict(
            allowed_border=-1,
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=True,
                min_pos_iou=0.3,
                neg_iou_thr=0.3,
                pos_iou_thr=0.7,
                type='MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=False,
                neg_pos_ub=-1,
                num=256,
                pos_fraction=0.5,
                type='RandomSampler')),
        rpn_proposal=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=2000)),
    type='RSMaskRCNN')
optim_wrapper = dict(
    optimizer=dict(lr=0.0001, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0))),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=100, start_factor=0.001, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=36,
        gamma=0.1,
        milestones=[
            24,
            33,
        ],
        type='MultiStepLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/instances_test.json',
        backend_args=None,
        data_prefix=dict(img='test/'),
        data_root='data/gravel_roboflow_414_mmdet/',
        metainfo=dict(classes=('gravel', ), palette=[
            (
                255,
                140,
                0,
            ),
        ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                640,
                640,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='GravelDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=False,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='data/gravel_roboflow_414_mmdet/annotations/instances_test.json',
    backend_args=None,
    format_only=False,
    metric='segm',
    type='FasterCocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        640,
        640,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(max_epochs=36, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=2,
    dataset=dict(
        ann_file='annotations/instances_train.json',
        backend_args=None,
        data_prefix=dict(img='train/'),
        data_root='data/gravel_roboflow_414_mmdet/',
        filter_cfg=dict(filter_empty_gt=False, min_size=1),
        metainfo=dict(classes=('gravel', ), palette=[
            (
                255,
                140,
                0,
            ),
        ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                keep_ratio=True,
                scales=[
                    (
                        512,
                        512,
                    ),
                    (
                        640,
                        640,
                    ),
                    (
                        768,
                        768,
                    ),
                ],
                type='RandomChoiceResize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackDetInputs'),
        ],
        type='GravelDataset'),
    num_workers=2,
    persistent_workers=False,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        keep_ratio=True,
        scales=[
            (
                512,
                512,
            ),
            (
                640,
                640,
            ),
            (
                768,
                768,
            ),
        ],
        type='RandomChoiceResize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/instances_val.json',
        backend_args=None,
        data_prefix=dict(img='val/'),
        data_root='data/gravel_roboflow_414_mmdet/',
        metainfo=dict(classes=('gravel', ), palette=[
            (
                255,
                140,
                0,
            ),
        ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                640,
                640,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='GravelDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=False,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='data/gravel_roboflow_414_mmdet/annotations/instances_val.json',
    backend_args=None,
    format_only=False,
    metric='segm',
    type='FasterCocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = '/home/yxy18034962/projects/mmdetection/work_dirs_gravel/mask_rcnn_rs_lightmamba_s4_global_attn_hf_fpn_36e_gravel_414/test_eval'

[Stage4-Attn] Replaced 2 VSS blocks in stages [3] with global attention.
Loaded official checkpoint: checkpoints/vmamba/vssm_tiny_0230_ckpt_epoch_262.pth
Loaded params: 14.494M / 36.660M (39.54%)
Exact params: 14.494M, Partial params: 0.000M, Skipped keys: 0
Missing keys: 127
Unexpected keys: 0
04/15 02:45:52 - mmengine - INFO - Distributed training is not used, all SyncBatchNorm (SyncBN) layers in the model will be automatically reverted to BatchNormXd layers if they are used.
04/15 02:45:52 - mmengine - INFO - Hooks will be executed in the following order:
before_run:
(VERY_HIGH   ) RuntimeInfoHook                    
(BELOW_NORMAL) LoggerHook                         
 -------------------- 
before_train:
(VERY_HIGH   ) RuntimeInfoHook                    
(NORMAL      ) IterTimerHook                      
(VERY_LOW    ) CheckpointHook                     
 -------------------- 
before_train_epoch:
(VERY_HIGH   ) RuntimeInfoHook                    
(NORMAL      ) IterTimerHook                      
(NORMAL      ) DistSamplerSeedHook                
 -------------------- 
before_train_iter:
(VERY_HIGH   ) RuntimeInfoHook                    
(NORMAL      ) IterTimerHook                      
 -------------------- 
after_train_iter:
(VERY_HIGH   ) RuntimeInfoHook                    
(NORMAL      ) IterTimerHook                      
(BELOW_NORMAL) LoggerHook                         
(LOW         ) ParamSchedulerHook                 
(VERY_LOW    ) CheckpointHook                     
 -------------------- 
after_train_epoch:
(NORMAL      ) IterTimerHook                      
(LOW         ) ParamSchedulerHook                 
(VERY_LOW    ) CheckpointHook                     
 -------------------- 
before_val:
(VERY_HIGH   ) RuntimeInfoHook                    
 -------------------- 
before_val_epoch:
(NORMAL      ) IterTimerHook                      
 -------------------- 
before_val_iter:
(NORMAL      ) IterTimerHook                      
 -------------------- 
after_val_iter:
(NORMAL      ) IterTimerHook                      
(NORMAL      ) DetVisualizationHook               
(BELOW_NORMAL) LoggerHook                         
 -------------------- 
after_val_epoch:
(VERY_HIGH   ) RuntimeInfoHook                    
(NORMAL      ) IterTimerHook                      
(BELOW_NORMAL) LoggerHook                         
(LOW         ) ParamSchedulerHook                 
(VERY_LOW    ) CheckpointHook                     
 -------------------- 
after_val:
(VERY_HIGH   ) RuntimeInfoHook                    
 -------------------- 
after_train:
(VERY_HIGH   ) RuntimeInfoHook                    
(VERY_LOW    ) CheckpointHook                     
 -------------------- 
before_test:
(VERY_HIGH   ) RuntimeInfoHook                    
 -------------------- 
before_test_epoch:
(NORMAL      ) IterTimerHook                      
 -------------------- 
before_test_iter:
(NORMAL      ) IterTimerHook                      
 -------------------- 
after_test_iter:
(NORMAL      ) IterTimerHook                      
(NORMAL      ) DetVisualizationHook               
(BELOW_NORMAL) LoggerHook                         
 -------------------- 
after_test_epoch:
(VERY_HIGH   ) RuntimeInfoHook                    
(NORMAL      ) IterTimerHook                      
(BELOW_NORMAL) LoggerHook                         
 -------------------- 
after_test:
(VERY_HIGH   ) RuntimeInfoHook                    
 -------------------- 
after_run:
(BELOW_NORMAL) LoggerHook                         
 -------------------- 
loading annotations into memory...
Done (t=0.12s)
creating index...
index created!
loading annotations into memory...
Done (t=0.02s)
creating index...
index created!
Loads checkpoint by local backend from path: /home/yxy18034962/projects/mmdetection/work_dirs_gravel/mask_rcnn_rs_lightmamba_s4_global_attn_hf_fpn_36e_gravel_414/best_coco_segm_mAP_epoch_26.pth
04/15 02:45:53 - mmengine - INFO - Load checkpoint from /home/yxy18034962/projects/mmdetection/work_dirs_gravel/mask_rcnn_rs_lightmamba_s4_global_attn_hf_fpn_36e_gravel_414/best_coco_segm_mAP_epoch_26.pth
04/15 02:45:59 - mmengine - INFO - Epoch(test) [50/50]    eta: 0:00:00  time: 0.1326  data_time: 0.0781  memory: 796  
04/15 02:46:00 - mmengine - INFO - Evaluating segm...
04/15 02:46:00 - mmengine - INFO - Evaluate annotation type *segm*
04/15 02:46:00 - mmengine - INFO - COCOeval_opt.evaluate() finished...
04/15 02:46:00 - mmengine - INFO - DONE (t=0.12s).
04/15 02:46:00 - mmengine - INFO - Accumulating evaluation results...
04/15 02:46:00 - mmengine - INFO - COCOeval_opt.accumulate() finished...
04/15 02:46:00 - mmengine - INFO - DONE (t=0.00s).
04/15 02:46:00 - mmengine - INFO -  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.276
04/15 02:46:00 - mmengine - INFO -  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.399
04/15 02:46:00 - mmengine - INFO -  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.326
04/15 02:46:00 - mmengine - INFO -  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.241
04/15 02:46:00 - mmengine - INFO -  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.600
04/15 02:46:00 - mmengine - INFO -  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.687
04/15 02:46:00 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.297
04/15 02:46:00 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.297
04/15 02:46:00 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.297
04/15 02:46:00 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.262
04/15 02:46:00 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.633
04/15 02:46:00 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.705
04/15 02:46:00 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.404
04/15 02:46:00 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.346
04/15 02:46:00 - mmengine - INFO - segm_mAP_copypaste: 0.276 0.399 0.326 0.241 0.600 0.687
04/15 02:46:00 - mmengine - INFO - Epoch(test) [50/50]    coco/segm_mAP: 0.2760  coco/segm_mAP_50: 0.3990  coco/segm_mAP_75: 0.3260  coco/segm_mAP_s: 0.2410  coco/segm_mAP_m: 0.6000  coco/segm_mAP_l: 0.6870  data_time: 0.0781  time: 0.1326
/home/yxy18034962/projects/mmdetection/work_dirs_gravel/mask_rcnn_rs_lightmamba_s4_global_attn_hf_fpn_36e_gravel_414/test_eval.log


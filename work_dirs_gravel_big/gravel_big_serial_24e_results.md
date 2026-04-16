# Gravel Big Serial 24e Results

该文件由串行训练脚本自动维护。

## Mask R-CNN R50

- config: configs/gravel_big/mask_rcnn_r50_fpn_24e_gravel_big.py
- work_dir: work_dirs_gravel_big/mask_rcnn_r50_fpn_24e_gravel_big
- best_ckpt: /home/yxy18034962/projects/mmdetection/work_dirs_gravel_big/mask_rcnn_r50_fpn_24e_gravel_big/best_coco_segm_mAP_epoch_17.pth
- eval_log: [2026-04-15 13:33:22] test eval: configs/gravel_big/mask_rcnn_r50_fpn_24e_gravel_big.py with /home/yxy18034962/projects/mmdetection/work_dirs_gravel_big/mask_rcnn_r50_fpn_24e_gravel_big/best_coco_segm_mAP_epoch_17.pth
04/15 13:33:25 - mmengine - INFO - 
------------------------------------------------------------
System environment:
    sys.platform: linux
    Python: 3.10.19 (main, Oct 21 2025, 16:43:05) [GCC 11.2.0]
    CUDA available: True
    MUSA available: False
    numpy_random_seed: 1425074247
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
    seed: 1425074247
    Distributed launcher: none
    Distributed training: False
    GPU number: 1
------------------------------------------------------------

04/15 13:33:25 - mmengine - INFO - Config:
auto_scale_lr = dict(base_batch_size=16, enable=False)
backend_args = None
data_root = 'data/gravel_big_mmdet/'
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
load_from = '/home/yxy18034962/projects/mmdetection/work_dirs_gravel_big/mask_rcnn_r50_fpn_24e_gravel_big/best_coco_segm_mAP_epoch_17.pth'
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
        depth=50,
        frozen_stages=1,
        init_cfg=dict(checkpoint='torchvision://resnet50', type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='pytorch',
        type='ResNet'),
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
            256,
            512,
            1024,
            2048,
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
    optimizer=dict(lr=0.005, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=100, start_factor=0.001, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=24,
        gamma=0.1,
        milestones=[
            16,
            22,
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
        data_root='data/gravel_big_mmdet/',
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
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='data/gravel_big_mmdet/annotations/instances_test.json',
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
train_cfg = dict(max_epochs=24, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=4,
    dataset=dict(
        ann_file='annotations/instances_train.json',
        backend_args=None,
        data_prefix=dict(img='train/'),
        data_root='data/gravel_big_mmdet/',
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
    num_workers=4,
    persistent_workers=True,
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
        data_root='data/gravel_big_mmdet/',
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
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='data/gravel_big_mmdet/annotations/instances_val.json',
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
work_dir = '/home/yxy18034962/projects/mmdetection/work_dirs_gravel_big/mask_rcnn_r50_fpn_24e_gravel_big/test_eval'

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
04/15 13:33:26 - mmengine - INFO - Distributed training is not used, all SyncBatchNorm (SyncBN) layers in the model will be automatically reverted to BatchNormXd layers if they are used.
04/15 13:33:26 - mmengine - INFO - Hooks will be executed in the following order:
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
Done (t=0.01s)
creating index...
index created!
loading annotations into memory...
Done (t=0.01s)
creating index...
index created!
Loads checkpoint by local backend from path: /home/yxy18034962/projects/mmdetection/work_dirs_gravel_big/mask_rcnn_r50_fpn_24e_gravel_big/best_coco_segm_mAP_epoch_17.pth
04/15 13:33:27 - mmengine - INFO - Load checkpoint from /home/yxy18034962/projects/mmdetection/work_dirs_gravel_big/mask_rcnn_r50_fpn_24e_gravel_big/best_coco_segm_mAP_epoch_17.pth
04/15 13:33:30 - mmengine - INFO - Epoch(test) [ 50/145]    eta: 0:00:05  time: 0.0552  data_time: 0.0210  memory: 567  
04/15 13:33:32 - mmengine - INFO - Epoch(test) [100/145]    eta: 0:00:02  time: 0.0391  data_time: 0.0158  memory: 566  
04/15 13:33:33 - mmengine - INFO - Evaluating segm...
04/15 13:33:33 - mmengine - INFO - Evaluate annotation type *segm*
04/15 13:33:33 - mmengine - INFO - COCOeval_opt.evaluate() finished...
04/15 13:33:33 - mmengine - INFO - DONE (t=0.09s).
04/15 13:33:33 - mmengine - INFO - Accumulating evaluation results...
04/15 13:33:33 - mmengine - INFO - COCOeval_opt.accumulate() finished...
04/15 13:33:33 - mmengine - INFO - DONE (t=0.00s).
04/15 13:33:33 - mmengine - INFO -  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.305
04/15 13:33:33 - mmengine - INFO -  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.586
04/15 13:33:33 - mmengine - INFO -  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.283
04/15 13:33:33 - mmengine - INFO -  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.157
04/15 13:33:33 - mmengine - INFO -  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.422
04/15 13:33:33 - mmengine - INFO -  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.677
04/15 13:33:33 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.393
04/15 13:33:33 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.393
04/15 13:33:33 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.393
04/15 13:33:33 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.269
04/15 13:33:33 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.512
04/15 13:33:33 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.714
04/15 13:33:33 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.692
04/15 13:33:33 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.394
04/15 13:33:33 - mmengine - INFO - segm_mAP_copypaste: 0.305 0.586 0.283 0.157 0.422 0.677
04/15 13:33:33 - mmengine - INFO - Epoch(test) [145/145]    coco/segm_mAP: 0.3050  coco/segm_mAP_50: 0.5860  coco/segm_mAP_75: 0.2830  coco/segm_mAP_s: 0.1570  coco/segm_mAP_m: 0.4220  coco/segm_mAP_l: 0.6770  data_time: 0.0156  time: 0.0420
/home/yxy18034962/projects/mmdetection/work_dirs_gravel_big/mask_rcnn_r50_fpn_24e_gravel_big/test_eval.log


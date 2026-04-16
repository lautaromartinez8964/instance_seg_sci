# gravel_roboflow 各模型指标汇总

说明：
- Params 为按实际配置构建后统计得到的检测器总参数量。
- FLOPs 统一按 gravel 推理配置 640×640 记录。
- FPS、latency、CUDA Memory 由 best checkpoint 实测：`inference_detector` 直接计时（warmup=20, iter=100, batch=1, 640×640 dummy image）。
- 精度指标默认记录 val set 结果。
- 对 VMamba 系列，FLOPs 使用 `tools/analysis_tools/get_flops_vmamba_safe.py` 同口径估算；由于 Triton / 自定义算子不可被完整 trace，数值应视为统一对比口径下的 safe 估算值。

## 当前对比总表

| 模型 | segm mAP | AP50 | AP75 | APs | APm | APl | Params (M) | FLOPs (G) | FPS | latency | CUDA Memory |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Mask R-CNN R50 | 0.284 | 0.416 | 0.341 | 0.249 | 0.600 | 0.622 | 43.971 | 134 | 70.1 img/s | 14.3 ms/img | 346.6 MB |
| Official VMamba 2292 | 0.294 | 0.416 | 0.353 | 0.260 | 0.615 | 0.619 | 57.564 | 126 | 44.7 img/s | 22.4 ms/img | 557.5 MB |
| RS-LightMamba S4 GlobalAttn | 0.294 | 0.416 | 0.352 | 0.260 | 0.617 | 0.613 | 56.507 | 126 | 44.5 img/s | 22.5 ms/img | 550.2 MB |
| RS-LightMamba S4 GlobalAttn HF-FPN | 0.294 | 0.417 | 0.353 | 0.259 | 0.619 | 0.610 | 56.507 | 126 | 43.8 img/s | 22.8 ms/img | 550.5 MB |

# Mask R-CNN R50 Baseline 完整指标记录

## 1. 精度指标（gravel_roboflow val set, 640×640）

### 实例分割 (segm)
| 指标 | 数值 |
|---|---|
| mAP | 0.284 |
| AP50 | 0.416 |
| AP75 | 0.341 |
| APs | 0.249 |
| APm | 0.600 |
| APl | 0.622 |

## 2. 模型参数指标
| 指标 | 数值 |
|---|---|
| Params (M) | 43.971 |
| FLOPs (G) | 134 |
| FPS | 70.1 img/s |
| latency | 14.3 ms/img |
| CUDA Memory | 346.6 MB |

## 3. Per-class AP (segm)
| 类别 | AP |
|---|---|
| gravel | 0.284 |

## 4. 训练配置
- Backbone: ResNet-50 + FPN
- Pretrained: checkpoints/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth
- Schedule: 36 epochs
- Optimizer: SGD, lr=0.0025, momentum=0.9, weight_decay=0.0001
- Batch size: 2
- Input size: train 为 512/640/768 多尺度随机缩放；val/test 为 640×640
- GPU: RTX 5090

## 5. 备注
- Best checkpoint: work_dirs_gravel/mask_rcnn_r50_fpn_36e_gravel_414/best_coco_segm_mAP_epoch_25.pth
- Best epoch: 25，best segm mAP = 0.284；epoch 36 结束时 segm mAP 仍为 0.284。
- 当前数据集为单类别实例分割，因此 per-class segm AP 与 overall segm mAP 一致。
- 训练中 COCO segm evaluator 曾因 pycocotools 在当前环境下不稳定而触发崩溃，最终采用 FasterCocoMetric 完成稳定评估与收敛记录。
- 定性可视化结果已导出到 outputs_gravel_roboflow/resnet50_1 下的 val/test 目录。


# Mask R-CNN Official VMamba 2292 完整指标记录

## 1. 精度指标（gravel_roboflow val set, 640×640）

### 实例分割 (segm)
| 指标 | 数值 |
|---|---|
| mAP | 0.294 |
| AP50 | 0.416 |
| AP75 | 0.353 |
| APs | 0.260 |
| APm | 0.615 |
| APl | 0.619 |

## 2. 模型参数指标
| 指标 | 数值 |
|---|---|
| Params (M) | 57.564 |
| FLOPs (G) | 126 |
| FPS | 44.7 img/s |
| latency | 22.4 ms/img |
| CUDA Memory | 557.5 MB |

## 3. Per-class AP (segm)
| 类别 | AP |
|---|---|
| gravel | 0.294 |

## 4. 训练配置
- Backbone: MM_VMamba（official VMamba，dims=[96,192,384,768], depths=[2,2,9,2]）
- Pretrained: checkpoints/vmamba/vssm_tiny_0230_ckpt_epoch_262.pth
- Schedule: 36 epochs
- Optimizer: AdamW, lr=1e-4, weight_decay=0.05
- Batch size: 2
- Input size: train 为 512/640/768 多尺度随机缩放；val/test 为 640×640
- GPU: RTX 5090

## 5. 备注
- Best checkpoint: work_dirs_gravel/mask_rcnn_vmamba_official_2292_fpn_36e_gravel_414/best_coco_segm_mAP_epoch_28.pth
- Best epoch: 28，best segm mAP = 0.294；相对 R50 baseline 提升 +0.010。
- 当前数据集为单类别实例分割，因此 per-class segm AP 与 overall segm mAP 一致。
- Params 为实际模型构建统计；FLOPs 为 VMamba-safe 估算值；FPS / latency / CUDA Memory 由 640×640 dummy image + `inference_detector` 统一实测。


# Mask R-CNN RS-LightMamba S4 GlobalAttn 完整指标记录

## 1. 精度指标（gravel_roboflow val set, 640×640）

### 实例分割 (segm)
| 指标 | 数值 |
|---|---|
| mAP | 0.294 |
| AP50 | 0.416 |
| AP75 | 0.352 |
| APs | 0.260 |
| APm | 0.617 |
| APl | 0.613 |

## 2. 模型参数指标
| 指标 | 数值 |
|---|---|
| Params (M) | 56.507 |
| FLOPs (G) | 126 |
| FPS | 44.5 img/s |
| latency | 22.5 ms/img |
| CUDA Memory | 550.2 MB |

## 3. Per-class AP (segm)
| 类别 | AP |
|---|---|
| gravel | 0.294 |

## 4. 训练配置
- Backbone: RSLightMambaBackbone（Stage 4 用 Global Attention 替换末层 VSS blocks）
- Detector: RSMaskRCNN + FPN
- Pretrained: checkpoints/vmamba/vssm_tiny_0230_ckpt_epoch_262.pth（official checkpoint partial load）
- Schedule: 36 epochs
- Optimizer: AdamW, lr=1e-4, weight_decay=0.05
- Batch size: 2
- Input size: train 为 512/640/768 多尺度随机缩放；val/test 为 640×640
- GPU: RTX 5090

## 5. 备注
- Best checkpoint: work_dirs_gravel/mask_rcnn_rs_lightmamba_s4_global_attn_fpn_36e_gravel_414/best_coco_segm_mAP_epoch_27.pth
- Best epoch: 27，best segm mAP = 0.294；与 official VMamba 2292 持平，AP75 -0.001，APm +0.002，APl -0.006。
- 当前数据集为单类别实例分割，因此 per-class segm AP 与 overall segm mAP 一致。
- Params 略少于 official 2292（-1.057M），速度基本持平；在 gravel_roboflow 上未观察到显著精度增益。


# Mask R-CNN RS-LightMamba S4 GlobalAttn HF-FPN 完整指标记录

## 1. 精度指标（gravel_roboflow val set, 640×640）

### 实例分割 (segm)
| 指标 | 数值 |
|---|---|
| mAP | 0.294 |
| AP50 | 0.417 |
| AP75 | 0.353 |
| APs | 0.259 |
| APm | 0.619 |
| APl | 0.610 |

## 2. 模型参数指标
| 指标 | 数值 |
|---|---|
| Params (M) | 56.507 |
| FLOPs (G) | 126 |
| FPS | 43.8 img/s |
| latency | 22.8 ms/img |
| CUDA Memory | 550.5 MB |

## 3. Per-class AP (segm)
| 类别 | AP |
|---|---|
| gravel | 0.294 |

## 4. 训练配置
- Backbone: RSLightMambaBackbone（Stage 4 Global Attention + output_hf_maps=True）
- Detector: RSMaskRCNN + HF_FPN
- 高频模块: 当前为拉普拉斯版 HF-FPN gravel 验证配置
- Pretrained: checkpoints/vmamba/vssm_tiny_0230_ckpt_epoch_262.pth（official checkpoint partial load）
- Schedule: 36 epochs
- Optimizer: AdamW, lr=1e-4, weight_decay=0.05
- Batch size: 2
- Input size: train 为 512/640/768 多尺度随机缩放；val/test 为 640×640
- GPU: RTX 5090

## 5. 备注
- Best checkpoint: work_dirs_gravel/mask_rcnn_rs_lightmamba_s4_global_attn_hf_fpn_36e_gravel_414/best_coco_segm_mAP_epoch_26.pth
- Best epoch: 26，best segm mAP = 0.294；与 official VMamba 2292 持平，AP50 +0.001，AP75 持平，APm +0.004，APl -0.009。
- 当前数据集为单类别实例分割，因此 per-class segm AP 与 overall segm mAP 一致。
- 相比 S4 GlobalAttn FPN，HF-FPN 在 gravel_roboflow 上未带来可分辨的总体 mAP 增益，且速度略降（44.5 → 43.8 img/s）。

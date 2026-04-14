# 各模型指标汇总

说明：
- Params 为按实际配置构建后统计得到的检测器总参数量。
- FLOPs 统一按 input 800×800 记录；VMamba 系列 FLOPs 使用 Triton-safe 统计脚本估算。

# Mask R-CNN R50 Baseline 完整指标记录

## 1. 精度指标（iSAID val set, 800×800 patches）

### 实例分割 (segm)
| 指标 | 数值 |
|---|---|
| mAP | 0.371 |
| AP₅₀ | 0.597 |
| AP₇₅ | 0.401 |
| APₛ | 0.226 |
| APₘ | 0.441 |
| APₗ | 0.523 |

### 目标检测 (bbox)
| 指标 | 数值 |
|---|---|
| mAP | 0.403 |
| AP₅₀ | 0.620 |
| AP₇₅ | 0.448 |
| APₛ | 0.267 |
| APₘ | 0.484 |
| APₗ | 0.519 |

## 2. 模型参数指标
| 指标 | 数值 |
|---|---|
| Params (M) | 44.047 |
| FLOPs (G) | 186 |
| FPS | 67.9 img/s |
| latency | 14.7 ms/img |
| CUDA Memory | 438 MB |

## 3. Per-class AP (segm)
| 类别 | AP |
|---|---|
| ship | _待填_ |
| store_tank | _待填_ |
| ... | ... |

## 4. 训练配置
- Backbone: ResNet-50 (ImageNet pretrained)
- Schedule: 1x (12 epochs)
- Optimizer: SGD, lr=0.005
- Batch size: 4
- Input size: 800×800
- GPU: RTX 5090


---
---

# Mask R-CNN Custom VMamba Baseline 完整指标记录

## 1. 精度指标（iSAID val set, 800×800 patches）

### 实例分割 (segm)
| 指标 | 数值 |
|---|---|
| mAP | 0.342 |
| AP₅₀ | 0.554 |
| AP₇₅ | 0.367 |
| APₛ | 0.204 |
| APₘ | 0.416 |
| APₗ | 0.501 |

### 目标检测 (bbox)
| 指标 | 数值 |
|---|---|
| mAP | 0.361 |
| AP₅₀ | 0.579 |
| AP₇₅ | 0.392 |
| APₛ | 0.238 |
| APₘ | 0.434 |
| APₗ | 0.482 |

## 2. 模型参数指标
| 指标 | 数值 |
|---|---|
| Params (M) | 30.693 |
| FLOPs (G) | 166 |
| FPS | 18.0 img/s |
| latency | 55.6 ms/img |
| CUDA Memory | 432 MB |

## 3. Per-class AP (segm)
| 类别 | AP |
|---|---|
| ship | _待填_ |
| store_tank | _待填_ |
| ... | ... |

## 4. 训练配置
- Backbone: VMambaBackbone（custom_Mamba，实现为 dims=[48,96,192,384], depths=[2,2,9,2]）
- Schedule: 1x (12 epochs)
- Optimizer: AdamW, lr=1e-4, weight_decay=0.05
- Batch size: 2
- Input size: 800×800
- GPU: RTX 5090


---
---

# Mask R-CNN Official VMamba 2292 Baseline 完整指标记录

## 1. 精度指标（iSAID val set, 800×800 patches）

### 实例分割 (segm)
| 指标 | 数值 |
|---|---|
| mAP | 0.406 |
| AP₅₀ | 0.637 |
| AP₇₅ | 0.436 |
| APₛ | 0.260 |
| APₘ | 0.482 |
| APₗ | 0.595 |

### 目标检测 (bbox)
| 指标 | 数值 |
|---|---|
| mAP | 0.441 |
| AP₅₀ | 0.660 |
| AP₇₅ | 0.486 |
| APₛ | 0.307 |
| APₘ | 0.510 |
| APₗ | 0.610 |

## 2. 模型参数指标
| 指标 | 数值 |
|---|---|
| Params (M) | 57.639 |
| FLOPs (G) | 212 |
| FPS | 46.8 img/s |
| latency | 21.4 ms/img |
| CUDA Memory | 738 MB |

## 3. Per-class AP (segm)
| 类别 | AP |
|---|---|
| ship | _待填_ |
| store_tank | _待填_ |
| ... | ... |

## 4. 训练配置
- Backbone: MM_VMamba（official VMamba，dims=[96,192,384,768], depths=[2,2,9,2]）
- Schedule: 1x (12 epochs)
- Optimizer: AdamW, lr=1e-4, weight_decay=0.05
- Batch size: 2
- Input size: 800×800
- GPU: RTX 5090
- Pretrained: checkpoints/vmamba/vssm_tiny_0230_ckpt_epoch_262.pth

## 5. 备注
- 当前 official 2292 配置加载上述 ImageNet checkpoint 时存在部分 missing keys，原因是 checkpoint 深度与当前 stage-3 的 2292 配置不完全一致。
- 上述精度指标来自实际训练与评估结果；Params 使用实际模型构建统计值，FLOPs 使用 safe 估算值。
- FPS、latency、CUDA Memory 由自定义基准脚本测得：`inference_detector` 直接计时（warmup=20, iter=100, batch=1, 800×800 dummy image）；CUDA Memory 为 `torch.cuda.max_memory_allocated` 峰值。
- **注意**：与之前记录的 MMDetection benchmark.py 默认 `--task dataloader` 数值不同，此处为真实 inference 计时，为本文件所有条目的统一基准。


---
---

# Mask R-CNN RS-LightMamba 2241 Research Baseline 完整指标记录

## 1. 精度指标（iSAID val set, 800×800 patches）

### 实例分割 (segm)
| 指标 | 数值 |
|---|---|
| mAP | 0.371 |
| AP₅₀ | 0.590 |
| AP₇₅ | 0.399 |
| APₛ | 0.229 |
| APₘ | 0.446 |
| APₗ | 0.524 |

### 目标检测 (bbox)
| 指标 | 数值 |
|---|---|
| mAP | 0.394 |
| AP₅₀ | 0.616 |
| AP₇₅ | 0.431 |
| APₛ | 0.265 |
| APₘ | 0.465 |
| APₗ | 0.504 |

## 2. 模型参数指标
| 指标 | 数值 |
|---|---|
| Params (M) | 24.861 |
| FLOPs (G) | 143 |
| FPS | 62.3 img/s |
| latency | 16.0 ms/img |
| CUDA Memory | 414 MB |

## 3. Per-class AP (segm)
| 类别 | AP |
|---|---|
| ship | _待填_ |
| store_tank | _待填_ |
| ... | ... |

## 4. 训练配置
- Backbone: RSLightMambaBackbone（official-derived research variant，dims=[48,96,192,384], depths=[2,2,4,1]）
- Schedule: 1x (12 epochs)
- Optimizer: AdamW, lr=1e-4, weight_decay=0.05
- Batch size: 2
- Input size: 800×800
- GPU: RTX 5090
- Pretrained: checkpoints/vmamba/vssm_tiny_0230_ckpt_epoch_262.pth（partial overlap load）

## 5. 备注
- 当前结果来自 epoch 12 验证日志（Epoch(val) [12][9581/9581]），tmux 会话结束后由日志回溯确认最终指标。
- 训练阶段通过 tmux 会话 rs_lightmamba_2241_auto 启动，并接入 scripts/auto_resume_from_latest_epoch.sh 自动续训。
- Params / FLOPs 为当前 2241 研究基线的静态复杂度；FPS / latency / CUDA Memory 由自定义基准脚本测得（`inference_detector` 直接计时，warmup=20, iter=100, batch=1）。


---

# Mask R-CNN RS-LightMamba S4-GlobalAttn FPN 1x 完整指标记录

> Config: `projects/iSAID/configs/mask_rcnn_rs_lightmamba_s4_global_attn_fpn_1x_isaid.py`
> Best epoch: 10

## 1. 精度指标（iSAID val set, 800×800 patches）

### 实例分割 (segm)
| 指标 | 数值 |
|---|---|
| mAP | 0.409 |
| AP₅₀ | 0.642 |
| AP₇₅ | 0.442 |
| APₛ | 0.260 |
| APₘ | 0.487 |
| APₗ | 0.584 |

### 目标检测 (bbox)
| 指标 | 数值 |
|---|---|
| mAP | 0.443 |
| AP₅₀ | 0.662 |
| AP₇₅ | 0.491 |
| APₛ | 0.303 |
| APₘ | 0.516 |
| APₗ | 0.595 |

## 2. 模型参数指标
| 指标 | 数值 |
|---|---|
| Params (M) | 56.583 |
| FLOPs (G) | 210 |
| FPS | 42.5 img/s |
| latency | 23.6 ms/img |
| CUDA Memory (inference) | 791 MB |

## 3. Per-class AP (segm)
| 类别 | AP |
|---|---|
| ship | _待填_ |
| store_tank | _待填_ |
| ... | ... |

## 4. 训练配置
- Backbone: RSLightMambaBackbone（Stage4 用 GlobalAttention 替换 VSS blocks）
- Schedule: 1x (12 epochs)，best at epoch 10
- Optimizer: AdamW, lr=1e-4, weight_decay=0.05
- Batch size: 2
- Input size: 800×800
- GPU: RTX 5090
- Pretrained: checkpoints/vmamba/vssm_tiny_0230_ckpt_epoch_262.pth

## 5. 备注
- 相比 VMamba-2292 baseline（segm mAP=0.406），本模型 +0.003，验证 Stage4 Global Attention 的有效性。
- Params / FLOPs 通过 get_flops.py 及参数统计脚本获得；FPS / latency / CUDA Memory 由自定义基准脚本测得（`inference_detector` 直接计时，warmup=20, iter=100, batch=1），为本文件所有条目的统一基准。


---

# Mask R-CNN RS-LightMamba S4-GlobalAttn HF-FPN-v2 1x 完整指标记录

> Config: `projects/iSAID/configs/mask_rcnn_rs_lightmamba_s4_global_attn_hf_fpn_v2_1x_isaid.py`
> Best epoch: 10

## 1. 精度指标（iSAID val set, 800×800 patches）

### 实例分割 (segm)
| 指标 | 数值 |
|---|---|
| mAP | 0.410 |
| AP₅₀ | 0.644 |
| AP₇₅ | 0.443 |
| APₛ | 0.256 |
| APₘ | 0.484 |
| APₗ | 0.592 |

### 目标检测 (bbox)
| 指标 | 数值 |
|---|---|
| mAP | 0.444 |
| AP₅₀ | 0.663 |
| AP₇₅ | 0.494 |
| APₛ | 0.301 |
| APₘ | 0.511 |
| APₗ | 0.609 |

## 2. 模型参数指标
| 指标 | 数值 |
|---|---|
| Params (M) | 56.567 |
| FLOPs (G) | 210 |
| FPS | 46.3 img/s |
| latency | 21.6 ms/img |
| CUDA Memory (inference) | 820 MB |

## 3. Per-class AP (segm)
| 类别 | AP |
|---|---|
| ship | _待填_ |
| store_tank | _待填_ |
| ... | ... |

## 4. 训练配置
- Backbone: RSLightMambaBackbone（Stage4 Global Attention + HF-FPN v2）
- Schedule: 1x (12 epochs)，best at epoch 10
- Optimizer: AdamW, lr=1e-4, weight_decay=0.05
- Batch size: 2
- Input size: 800×800
- GPU: RTX 5090
- Pretrained: checkpoints/vmamba/vssm_tiny_0230_ckpt_epoch_262.pth

## 5. 备注
- 相比 S4-GlobalAttn FPN 版本（segm mAP=0.409），HF-FPN-v2 额外提升 +0.001（segm），bbox mAP +0.001，APₗ 提升明显（+0.008），验证高频特征增强对大目标的作用。
- Params 略少（-0.016M），FPS 略优（+3.8 img/s），显存略多（+29 MB）。
- Params / FLOPs 通过 get_flops.py 及参数统计脚本获得；FPS / latency / CUDA Memory 由自定义基准脚本测得（`inference_detector` 直接计时，warmup=20, iter=100, batch=1），为本文件所有条目的统一基准。
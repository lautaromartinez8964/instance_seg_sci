# Gravel Big 各模型精度指标

## 统计口径

- 本页统一使用 36e 训练过程中的验证集 `segm` 指标，来源均为各自 `vis_data/scalars.json`。
- 记录每条实验线的最佳 epoch 指标、`epoch 36` 指标，以及 `best -> final` 回落幅度。
- 当前只保留三条主线：`Mask R-CNN R50`、`Official VMamba 2292`、`Shared DT-FPN v2`。
- 为保持口径一致，本页暂不混入 IBD 系列，也不把 `test_eval` 结果与训练期验证结果混写在同一张主表里。
- Params 为按实际配置构建后统计得到的检测器总参数量；FLOPs 统一按 input 640×640 记录，VMamba 系列使用 Triton-safe 统计脚本估算；FPS / latency / CUDA Memory 由 `profile_gravel_big.py` 测得（`inference_detector` 直接计时，warmup=20, iter=100, batch=1）。

---

## Mask R-CNN R50 完整指标记录

### 1. 精度指标（gravel_big val set, 640×640 patches）

| 指标 | 数值 |
| --- | ---: |
| segm mAP | 0.277 |
| mAP₅₀ | 0.436 |
| mAP₇₅ | 0.318 |
| mAPₛ | 0.214 |
| mAPₘ | 0.548 |
| mAPₗ | 0.672 |

### 2. 模型参数指标

| 指标 | 数值 |
| --- | ---: |
| Params (M) | 43.971 |
| FLOPs (G) | 142 |
| FPS | 85.1 img/s |
| latency | 11.7 ms/img |
| CUDA Memory | 487 MB |

### 3. 训练配置

- Backbone: ResNet-50 (ImageNet pretrained, COCO init)
- Schedule: 36e, milestones=[24, 33]
- Optimizer: SGD, lr=0.005, momentum=0.9, weight_decay=0.0001
- Batch size: 4
- Input size: 640×640 (multi-scale train: 512/640/768)
- GPU: RTX 5090
- Best epoch: 31
- epoch36 mAP: 0.276, best-final drop: 0.001

### 4. 指标来源

- 精度: `work_dirs_gravel_big/mask_rcnn_r50_fpn_36e_gravel_big/20260415_182802/vis_data/scalars.json`
- Params/FLOPs/FPS/Memory: `python tools/analysis_tools/profile_gravel_big.py configs/gravel_big/mask_rcnn_r50_fpn_36e_gravel_big.py --input-size 640 640`

---

## Official VMamba 2292 完整指标记录

### 1. 精度指标（gravel_big val set, 640×640 patches）

| 指标 | 数值 |
| --- | ---: |
| segm mAP | 0.283 |
| mAP₅₀ | 0.436 |
| mAP₇₅ | 0.327 |
| mAPₛ | 0.220 |
| mAPₘ | 0.552 |
| mAPₗ | 0.669 |

### 2. 模型参数指标

| 指标 | 数值 |
| --- | ---: |
| Params (M) | 57.564 |
| FLOPs (G) | 159 |
| FPS | 49.0 img/s |
| latency | 20.4 ms/img |
| CUDA Memory | 559 MB |

### 3. 训练配置

- Backbone: MM_VMamba (dims=[96,192,384,768], depths=[2,2,9,2])
- Schedule: 36e, milestones=[24, 33]
- Optimizer: AdamW, lr=0.0002, weight_decay=0.05
- Batch size: 4
- Input size: 640×640 (multi-scale train: 512/640/768)
- GPU: RTX 5090
- Pretrained: checkpoints/vmamba/vssm_tiny_0230_ckpt_epoch_262.pth
- Best epoch: 27
- epoch36 mAP: 0.281, best-final drop: 0.002

### 4. 备注

- 加载 ImageNet checkpoint 时存在部分 missing keys（checkpoint 深度与 stage-3 的 2292 配置不完全一致）。
- FLOPs 使用 `get_flops_vmamba_safe.py` (Triton-safe) 估算，backbone FLOPs 由 `Backbone_VSSM.flops()` 解析计算，Neck+Head FLOPs 由 mmengine `get_model_complexity_info` 追踪。

### 5. 指标来源

- 精度: `work_dirs_gravel_big/mask_rcnn_vmamba_official_2292_fpn_36e_gravel_big/20260415_213016/vis_data/scalars.json`
- Params/FLOPs: `python tools/analysis_tools/get_flops_vmamba_safe.py configs/gravel_big/mask_rcnn_vmamba_official_2292_fpn_36e_gravel_big.py`
- FPS/Memory: `python tools/analysis_tools/profile_gravel_big.py configs/gravel_big/mask_rcnn_vmamba_official_2292_fpn_36e_gravel_big.py --input-size 640 640`

---

## Shared DT-FPN v2 完整指标记录

### 1. 精度指标（gravel_big val set, 640×640 patches）

| 指标 | 数值 |
| --- | ---: |
| segm mAP | 0.286 |
| mAP₅₀ | 0.437 |
| mAP₇₅ | 0.331 |
| mAPₛ | 0.225 |
| mAPₘ | 0.555 |
| mAPₗ | 0.673 |

### 2. 模型参数指标

| 指标 | 数值 |
| --- | ---: |
| Params (M) | 60.072 |
| FLOPs (G) | 184 |
| FPS | 44.8 img/s |
| latency | 22.3 ms/img |
| CUDA Memory | 572 MB |

### 3. 训练配置

- Backbone: MM_VMamba (dims=[96,192,384,768], depths=[2,2,9,2])，同 Official VMamba 2292
- Neck: DTFPN (guided_levels=[0,1,2], dt_head_channels=128, dt_loss_weight=0.2)
- Detector: RSMaskRCNN (支持 DT 辅助 loss)
- Schedule: 36e
- Optimizer: AmpOptimWrapper (AdamW, lr=0.0002, weight_decay=0.05)
- Batch size: 4
- Input size: 640×640 (multi-scale train: 512/640/704)
- GPU: RTX 5090
- Pretrained: checkpoints/vmamba/vssm_tiny_0230_ckpt_epoch_262.pth
- Best epoch: 18
- epoch36 mAP: 0.278, best-final drop: 0.008

### 4. 备注

- 相比 Official VMamba 2292 基线（segm mAP=0.283），DT-FPN v2 提升 +0.003，且收敛更快（达峰 epoch 18 vs 27）。
- 但后期回落较大（best-final drop 0.008），暴露了 shared DT 设计的后期尺度冲突。
- 额外参数量约 2.5M（主要来自 DT 解码头），额外 FLOPs 约 25G。

### 5. 指标来源

- 精度: `work_dirs_gravel_big/mask_rcnn_gravel_lightmamba_dt_fpn_v2_36e_gravel_big/20260420_171117/vis_data/scalars.json`
- Params/FLOPs: `python tools/analysis_tools/get_flops_vmamba_safe.py configs/gravel_big/mask_rcnn_gravel_lightmamba_dt_fpn_v2_36e_gravel_big.py`
- FPS/Memory: `python tools/analysis_tools/profile_gravel_big.py configs/gravel_big/mask_rcnn_gravel_lightmamba_dt_fpn_v2_36e_gravel_big.py --input-size 640 640`

---

## 模型对比总表

### 精度对比

| 模型 | best epoch | segm mAP | mAP₅₀ | mAP₇₅ | mAPₛ | mAPₘ | mAPₗ | epoch36 mAP | best-final drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mask R-CNN R50 | 31 | 0.277 | 0.436 | 0.318 | 0.214 | 0.548 | 0.672 | 0.276 | 0.001 |
| Official VMamba 2292 | 27 | 0.283 | 0.436 | 0.327 | 0.220 | 0.552 | 0.669 | 0.281 | 0.002 |
| Shared DT-FPN v2 | 18 | 0.286 | 0.437 | 0.331 | 0.225 | 0.555 | 0.673 | 0.278 | 0.008 |

### 模型参数对比

| 模型 | Params (M) | FLOPs (G) | FPS (img/s) | Latency (ms) | CUDA Memory (MB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mask R-CNN R50 | 43.971 | 142 | 85.1 | 11.7 | 487 |
| Official VMamba 2292 | 57.564 | 159 | 49.0 | 20.4 | 559 |
| Shared DT-FPN v2 | 60.072 | 184 | 44.8 | 22.3 | 572 |

### 收敛速度对比

| 模型 | epoch10 mAP | epoch18 mAP | 达峰 epoch | 现象 |
| --- | ---: | ---: | ---: | --- |
| Mask R-CNN R50 | 0.264 | 0.273 | 31 | 前期稳定，但爬升偏慢，后半程才摸到最优。 |
| Official VMamba 2292 | 0.271 | 0.278 | 27 | 中前期已经优于 R50，后期还能继续小幅抬升。 |
| Shared DT-FPN v2 | 0.278 | 0.286 | 18 | 三者中收敛最快，早期直接冲顶，但后期回落最大。 |

---

## 关键结论

1. 当前三条主线里，峰值最高的是 `Shared DT-FPN v2`，最佳 `segm mAP = 0.286`。
2. 当前最强的纯 backbone/FPN 基线仍然是 `Official VMamba 2292`，最佳 `segm mAP = 0.283`。
3. `Mask R-CNN R50` 仍然是最稳的基础对照组，峰值不高，但 `best-final drop` 只有 `0.001`。
4. `Shared DT-FPN v2` 的最大价值在于两点同时成立：
	- 它把前期收敛速度拉到了三者最高。
	- 它也暴露了 shared DT 设计的后期尺度冲突，为后续 per-level DT 改造提供了最直接证据。
5. 效率对比：R50 在所有效率指标上遥遥领先（85.1 FPS, 487MB），VMamba 2292 较 R50 慢约 1.7×，DT-FPN v2 因额外 DT 解码头略慢于 2292（44.8 vs 49.0 FPS），额外开销约 9%。

---

## 当前排序

按验证集最佳 `segm mAP` 排序：

1. Shared DT-FPN v2: `0.286 @ epoch 18`
2. Official VMamba 2292: `0.283 @ epoch 27`
3. Mask R-CNN R50: `0.277 @ epoch 31`

---

这页文档后续如果要继续扩展，优先补：

- `Per-level DT-FPN v3`
- `DT-FPN v2` 的独立 `test_eval`

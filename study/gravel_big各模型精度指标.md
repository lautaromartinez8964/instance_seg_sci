# Gravel Big 各模型精度指标

## 统计口径

- 当前页面只保留两条有效主线：`RTMDet-Ins-Tiny baseline` 与 `RTMDet-Ins-Tiny Shared DT-FPN v3`。
- 精度统一采用 `gravel_big val set` 上的 `segm` 验证指标，来源为训练日志中的 `Epoch(val)` 最佳记录与最终 `epoch 120` 记录。
- Params / FLOPs / FPS / latency / CUDA Memory 统一按 `input 640x640` 统计。
- Params 与 FLOPs 来自实际配置构建后的 profiling；FPS / latency / CUDA Memory 由 `tools/analysis_tools/profile_gravel_big.py` 测得，口径为 `warmup=20, iter=100, batch=1`。
- 其余历史实验线均作废，不再纳入主表。

---

## RTMDet-Ins-Tiny baseline 完整指标记录

### 1. 精度指标（gravel_big val set, 640×640 patches）

| 指标 | 数值 |
| --- | ---: |
| segm mAP | 0.272 |
| mAP₅₀ | 0.430 |
| mAP₇₅ | 0.314 |
| mAPₛ | 0.213 |
| mAPₘ | 0.531 |
| mAPₗ | 0.658 |

### 2. 模型效率指标

| 指标 | 数值 |
| --- | ---: |
| Params (M) | 5.615 |
| FLOPs (G) | 11.873 |
| FPS | 156.7 img/s |
| latency | 6.4 ms/img |
| CUDA Memory | 64 MB |

### 3. 训练配置

- Backbone: CSPNeXt-Tiny (`deepen_factor=0.167`, `widen_factor=0.375`)
- Neck: CSPNeXtPAFPN
- Head: RTMDetInsSepBNHead
- Schedule: 120e
- Optimizer: AdamW, lr=2.5e-4, weight_decay=0.05
- Batch size: 16
- Input size: 640×640
- 强增强策略: 前 100e 使用 Mosaic + MixUp，后 20e 通过 `PipelineSwitchHook` 切换到弱增强精调
- GPU: RTX 5090
- Pretrained: `cspnext-tiny_imagenet_600e.pth`
- Best epoch: 119
- epoch120 mAP: 0.272
- best-final drop: 0.000

### 4. 备注

- 这是当前 RTMDet-Ins 路线的标准基线。
- 最优点出现在 119 epoch，但 120 epoch 基本持平，说明后期已经进入稳定平台。

### 5. 指标来源

- 精度: `work_dirs_gravel_big/rtmdet_ins_tiny_baseline_120e_gravel_big/20260424_122434/20260424_122434.log`
- Params / FLOPs / FPS / Memory: `python tools/analysis_tools/profile_gravel_big.py configs/gravel_big/rtmdet_ins_tiny_baseline_120e_gravel_big.py --input-size 640 640`

---

## RTMDet-Ins-Tiny Shared DT-FPN v3 完整指标记录

### 1. 精度指标（gravel_big val set, 640×640 patches）

| 指标 | 数值 |
| --- | ---: |
| segm mAP | 0.281 |
| mAP₅₀ | 0.443 |
| mAP₇₅ | 0.321 |
| mAPₛ | 0.218 |
| mAPₘ | 0.554 |
| mAPₗ | 0.676 |

### 2. 模型效率指标

| 指标 | 数值 |
| --- | ---: |
| Params (M) | 7.275 |
| FLOPs (G) | 15.989 |
| FPS | 155.1 img/s |
| latency | 6.4 ms/img |
| CUDA Memory | 70 MB |

### 3. 训练配置

- Backbone: CSPNeXt-Tiny (`deepen_factor=0.167`, `widen_factor=0.375`)，与 baseline 相同
- Neck: `DTCSPNeXtPAFPN`
- Detector: `RTMDetWithAuxNeck`
- Schedule: 120e
- Optimizer: AdamW, lr=2.5e-4, weight_decay=0.05
- Batch size: 16
- Input size: 640×640
- DT 设置: `guided_levels=[0,1]`, `dt_mode='shared'`, `dt_decoder_source='inputs'`, `dt_head_channels=128`, `dt_loss_weight=0.2`, `skip_csp_on_guided=True`
- DT 标签路径: `data/gravel_big_mmdet/auxiliary_labels_coreband/<split>/distance_transform/*.png`
- GPU: RTX 5090
- Best epoch: 111
- epoch120 mAP: 0.280
- best-final drop: 0.001

### 4. 备注

- 相比 baseline，最佳 `segm mAP` 提升 `+0.009`。
- 增益主要体现在中大尺度实例：`mAPₘ +0.023`，`mAPₗ +0.018`。
- 额外代价主要集中在 neck：总参数增加约 `+1.660M`，FLOPs 增加约 `+4.116G`。
- FPS 基本与 baseline 持平，但显存占用略高。
- 需要强调的是，这个版本的收益并非“纯 neck 改进”单因素结论；它同时改动了训练 pipeline，去掉了 baseline 的 Mosaic / MixUp + PipelineSwitch 方案，改成了全程简化增强并加入 DT 辅助监督。

### 5. 指标来源

- 精度: `work_dirs_gravel_big/rtmdet_ins_tiny_shared_dt_fpn_v3_120e_gravel_big/auto_logs/run_20260429_161133.log`
- Params / FLOPs / FPS / Memory: `python tools/analysis_tools/profile_gravel_big.py configs/gravel_big/rtmdet_ins_tiny_shared_dt_fpn_v3_120e_gravel_big.py --input-size 640 640`

---

## 模型对比总表

### 精度对比

| 模型 | best epoch | segm mAP | mAP₅₀ | mAP₇₅ | mAPₛ | mAPₘ | mAPₗ | epoch120 mAP | best-final drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RTMDet-Ins-Tiny baseline | 119 | 0.272 | 0.430 | 0.314 | 0.213 | 0.531 | 0.658 | 0.272 | 0.000 |
| RTMDet-Ins-Tiny Shared DT-FPN v3 | 111 | 0.281 | 0.443 | 0.321 | 0.218 | 0.554 | 0.676 | 0.280 | 0.001 |

### 模型效率对比

| 模型 | Params (M) | FLOPs (G) | FPS (img/s) | Latency (ms) | CUDA Memory (MB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| RTMDet-Ins-Tiny baseline | 5.615 | 11.873 | 156.7 | 6.4 | 64 |
| RTMDet-Ins-Tiny Shared DT-FPN v3 | 7.275 | 15.989 | 155.1 | 6.4 | 70 |

### 相对 baseline 的增量

| 模型 | segm mAP 增量 | Params 增量 | FLOPs 增量 | FPS 变化 | CUDA Memory 增量 |
| --- | ---: | ---: | ---: | ---: | ---: |
| RTMDet-Ins-Tiny Shared DT-FPN v3 | +0.009 | +1.660 M (+29.6%) | +4.116 G (+34.7%) | -1.6 img/s | +6 MB |

---

## 关键结论

1. 当前有效 RTMDet-Ins 路线里，峰值最高的是 `RTMDet-Ins-Tiny Shared DT-FPN v3`，最佳 `segm mAP = 0.281 @ epoch 111`。
2. baseline 的最佳 `segm mAP = 0.272`，且在 119 / 120 epoch 基本持平，说明后期训练非常稳定。
3. DT-FPN v3 的主要收益不在“小目标暴涨”，而在中大尺度实例的结构完整性与分离能力增强：`mAPₘ` 和 `mAPₗ` 提升最明显。
4. DT-FPN v3 的代价是约 `29.6%` 参数增长与 `34.7%` FLOPs 增长，但推理速度几乎不掉，说明这次增参集中在 neck 内部、且工程代价可控。
5. 从实验叙事上，应将该版本描述为“DT-guided neck + 简化训练增强策略”的联动收益，而不是纯 neck 单因素结论。

---

## 当前排序

按验证集最佳 `segm mAP` 排序：

1. RTMDet-Ins-Tiny Shared DT-FPN v3: `0.281 @ epoch 111`
2. RTMDet-Ins-Tiny baseline: `0.272 @ epoch 119`

---

这页文档后续如果继续扩展，优先补：

- `DT-FPN v3` 的严格纯结构对照实验
- `RTMDet-Ins-Tiny baseline / DT-FPN v3` 的 test set 独立评估结果

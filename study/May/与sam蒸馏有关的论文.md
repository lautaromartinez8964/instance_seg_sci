📚 2024-2025 高相关度论文精选（7篇）
1️⃣ EdgeTAM: On-Device Track Anything Model
arXiv: 2501.07256 | 时间: 2025.01 | 来源: Meta AI / NTU
代码: github.com/facebookresearch/EdgeTAM
① 核心问题：前人轻量化SAM只压缩image encoder，但SAM2新增的memory attention block才是视频分割的延迟瓶颈。

② 方法创新：

提出 2D Spatial Perceiver——把密集存储的帧级记忆用轻量Transformer + 一组可学习的query编码
将query分为global-level + patch-level两组，保持视频密集预测任务的空间结构
蒸馏pipeline（无推理开销）
③ 实验亮点：

DAVIS 2017: 87.7 J&F | MOSE: 70.0 | SA-V val: 72.3
iPhone 15 Pro Max 上跑 16 FPS！
💡 对你启发最大：2D Spatial Perceiver 的思路完全可以用到你的砾石多尺度特征融合中——把多帧/多尺度的砾石特征编码为一组可学习query，保持空间结构不丢失。

2️⃣ PicoSAM2: Low-Latency Segmentation In-Sensor
arXiv: 2506.11399 (检索到的条目名) | 时间: 2025.06 → 2025.11
作者: Bonazzi, Farronato, Qin 等 (ETH Zurich)
① 核心问题：在传感器内 (in-sensor) 实现实时分割——功耗/延迟/隐私三重约束下的极端轻量化。

② 方法创新：

Depthwise Separable U-Net 作为backbone（仅1.3M参数，336M MACs）
Knowledge Distillation 从SAM2蒸馏知识到极小的学生网络
优化到能在 Sony IMX500 智能传感器上运行
③ 实验亮点：

参数：1.3M（比你要求的十几M还轻！）
支持Sony IMX500边缘传感器执行
被后续 "Performance Analysis of Edge and In-Sensor AI Processors" 作为benchmark模型
💡 对你启发最大：Depthwise Sep U-Net 这个架构极其适合你的砾石分割——轻量、保持空间分辨率、GPU友好。你可以用它替换掉SAM的heavy image encoder。

3️⃣ EfficientSAM3: Progressive Hierarchical Distillation
arXiv: 2025.11 | 代码: github.com/SimonZeng7108/efficientsam3
① 核心问题：SAM3统一架构（共享vision backbone + DETR检测器 + 密集记忆追踪器）对端侧设备来说太重。

② 方法创新：

Progressive Hierarchical Distillation (PHD)：三段式蒸馏
Stage 1: Encoder Distillation (image encoder蒸馏)
Stage 2: Detector Distillation (DETR-style检测器蒸馏)
Stage 3: Tracker Distillation (记忆追踪器蒸馏)
从SAM1/2/3 多源知识渐进蒸馏到轻量学生
③ 实验亮点：

整个EfficientSAM3家族覆盖不同效率-精度trade-off
开源的蒸馏pipeline完整
💡 对你启发最大：PHD渐进蒸馏——你的砾石分割也可以这样做：先蒸image encoder，再蒸检测头/分割头。尤其是你如果使用SAM2作为teacher，完全可以分阶段蒸馏。

4️⃣ Tiny-YOLOSAM: Fast Hybrid Image Segmentation
arXiv: 2512.22193 | 时间: 2025.12 | 代码: github.com/Kenneth-Xu11566/tiny-yolosam
① 核心问题：TinySAM的"segment-everything"模式仍需要数百个prompt，速度慢。

② 方法创新：

YOLOv12检测器生成box prompt给TinySAM
未被YOLO覆盖的区域用稀疏点prompt补充
检测器指导prompt + 针对性稀疏采样 = 替代密集"segment-everything"
③ 实验亮点：

AR从 16.4% → 77.1%（mIoU: 19.2% → 67.8%）
端到端推理从 49.20s → 10.39s/image（4.7x加速）在Apple M1 Pro CPU上
💡 对你启发最大：检测器引导的prompt策略——你的砾石实例分割可以先接一个轻量检测器(YOLOv11-nano/RT-DETR-nano)先框出砾石区域，再在框内做精细分割，大幅减少全图密集prompt的计算量。

5️⃣ De-LightSAM: Modality-Decoupled Lightweight SAM
arXiv: 2407.01207(v1) → 2025.07更新 | 时间: 2024.07 → 2025.07
① 核心问题：医学图像多模态场景下，SAM无法很好地泛化到不同成像模态（CT/MRI/US等）。

② 方法创新：

Modality-Decoupled 架构：把模态特定特征和共享语义特征解耦
轻量学生网络 + 知识蒸馏
实现跨模态域泛化
③ 实验亮点：

在多个医学分割数据集上实现域泛化SOTA
参数远小于原始SAM，保持95%+性能
💡 对你启发最大：模态解耦思路应用到你的场景——砾石在不同光照/尺度/堆叠密度下视觉特征差异极大，你可以设计 "采集条件解耦" 的轻量模块，让模型学会区分"砾石本身的几何特征"和"拍摄条件导致的视觉偏差"。

6️⃣ UniUltra: Interactive Parameter-Efficient SAM2 for Universal Ultrasound Segmentation
arXiv: 2025.11
① 核心问题：SAM2在超声图像（域差异大）上性能退化严重。

② 方法创新：

参数高效微调（PEFT）风格适配SAM2
引入轻量adapter模块，只训练极少量参数
③ 实验亮点：

只用不到1%的可训练参数即实现超声域SOTA
💡 对你启发最大：PEFT + adapter 思路——你可以在轻量backbone中插入几个砾石场景特化的adapter层，仅在砾石数据上微调这些adapter，保持base模型泛化性。

7️⃣ Learning from Noisy Prompts: Saliency-Guided Prompt Distillation
arXiv: 2026.04 | 时间: 2026.04
① 核心问题：SAM依赖精确prompt，实际应用中prompt往往有噪声。

② 方法创新：

显著性引导的prompt蒸馏
学生网络学会从噪声prompt中恢复clean分割
③ 实验亮点：

在噪声prompt场景下性能显著优于直接使用SAM
💡 对你启发最大：prompt鲁棒性——你的砾石分割如果是"点击/框选"方式交互，实际使用中用户框选往往不准。你可以设计一个轻量的prompt去噪/细化模块。

⭐ 魔改 IDEA：轻量化密集砾石实例分割
总体定位：以 PicoSAM2 的 Depthwise Sep U-Net 为基础骨架，结合 EfficientSAM3 的渐进式蒸馏，最后用 检测器引导的prompt策略 提高密集场景下的分割效率。

Idea 1️⃣：砾石场景特化的渐进式多级蒸馏
技术路线：

Teacher: SAM2-Hiera-Large
    ↓ Stage 1: Encoder KD
学生编码器 ← MSE + Feature-level KL散度
    ↓ Stage 2: Decoder/Mask KD 
学生解码器 ← Mask Dice/BCE + Boundary-Aware Loss
    ↓ Stage 3: Instance-level KD
学生实例头 ← MaskIoU + 检测蒸馏
Backbone选用 深度可分离卷积 + 简化的Hierarchical Transformer（参考PicoSAM2的1.3M设计，放大到~5M）
蒸馏loss加入 砾石边界感知权重（砾石边缘分割是本任务的核心难点）
潜在贡献点：

首次在砾石/地质材料场景上验证SAM知识蒸馏的有效性
边界感知蒸馏loss的提出
实验设计：

消融三阶段蒸馏各自的贡献
对比：直接训练 vs PicoSAM2 vs EdgeTAM vs 你的方法
Idea 2️⃣：检测器引导的稀疏prompt + 稠密砾石场景适配
技术路线：

输入图像
    ↓
Ultra-Light Detector (YOLOv11-nano / RT-DETR-nano, <2M)
    ↓ 输出砾石候选框 + 置信度
自适应prompt生成器
    ├── 高置信度框 → 直接作为Box Prompt
    └── 低置信度/遮挡区域 → 按砾石密度分布采样Point Prompt
    ↓
轻量Mask Decoder（PicoSAM2风格）
    ↓ 
最终实例掩码
潜在贡献点：

解决密集砾石的堆叠遮挡 + 相互粘连问题
检测器+分割器的联合轻量化方案
实验设计：

评测全图推理 vs 检测引导推理的mAP@50:95 + FPS
不同detector backbone (YOLO-nano / RT-DETR-T / MobileNet-SSD) 对比
Idea 3️⃣：多尺度砾石感知的 2D Spatial Perceiver 特征增强
技术路线（借鉴EdgeTAM的2D Spatial Perceiver + 你自己的多尺度需求）：

轻量Backbone输出多尺度特征
        ↓
2D Spatial Perceiver (N=64 learnable queries per scale)
    ├── Global Queries (8个): 编码全局砾石分布/密度
    ├── Fine-Grained Queries (56个): 编码局部砾石边缘/纹理
    └── 保持2D空间排列 → 输出空间结构不乱
        ↓
Cross-Attention融合多尺度Perceiver输出
        ↓
Mask Decoder
潜在贡献点：

将EdgeTAM的视频记忆感知迁移到多尺度空间感知
解决了砾石在不同距离/缩放下的尺度冲突问题
实验设计：

Query分组数量消融：Global:Fine比例
对比：FPN / PAN / NAS-FPN vs 你的2D Perceiver
在砾石密集/稀疏/不同尺寸分布情况下的分项mAP
Idea 4️⃣：条件解耦 + 域不变特征学习（De-LightSAM启发）
技术路线：

输入图像
    ↓
Dual-Branch轻量Encoder
    ├── 条件分支（Lightweight）：学习光照/尺度/重叠度的域编码
    └── 内容分支（Shared）：学习砾石不变几何特征
        ↓
AdaIN / Feature Modulation融合
    ↓
Mask Decoder
潜在贡献点：

首次在遥感地质材料分割中提出"采集条件解耦"范式
泛化到不同砾石数据集（不同相机/光照/GPS位置）
实验设计：

条件分支添加/移除消融
跨数据集的zero-shot泛化测试
在极端光照/高密度堆叠条件下的鲁棒性
Idea 5️⃣（综合路线）："砾石SAM2-Lite" 整体框架推荐
我强烈建议师弟你走这条路线——它结合了上面4个idea的优点且工程上最可控：

                                  ┌─────────────────────┐
                                  │   SAM2 Teacher       │
                                  │  (Hiera-L / Hiera-B) │
                                  └─────────┬───────────┘
                                            │ 离线生成伪标签
                                            ▼
┌──────┐    ┌───────────────────────┐    ┌──────────┐
│ Input│───▶│ 深度可分离 ConvNeXt   │───▶│ 2D Spatial│
│ 图像  │    │  (PicoSAM2风格, ~5M)  │    │ Perceiver │
└──────┘    └───────────────────────┘    └─────┬────┘
                                               │
┌──────────────────┐                           │
│YOLOv11-nano检测器├───── box prompts ────────▶│
└──────────────────┘                           │
                                               ▼
                                      ┌──────────────────┐
                                      │ 轻量 Mask Decoder │
                                      │ (3-stage蒸馏训练) │
                                      └────────┬─────────┘
                                               ▼
                                      ┌──────────────────┐
                                      │ 砾石实例掩码输出  │
                                      │ (≤10M参数,手机端) │
                                      └──────────────────┘
argue 点：

轻量Backbone: 用深度可分离ConvNeXt替代传统heavy backbone（≈5M params）
2D Spatial Perceiver: 解决多尺度砾石特征融合（从EdgeTAM改造而来）
检测器引导prompt: YOLOv11-nano（~2M）+ 稀疏点prompt补充，解决密集场景漏检
三阶段蒸馏: 参照EfficientSAM3的PHD方案，从SAM2-Hiera-L蒸馏
总参数量: ~8-10M，在手机端可以跑实时（参考EdgeTAM的16 FPS on iPhone）
实验设计大纲：

实验	内容	预期结论
E1	全量vs蒸馏训练	蒸馏>直接训练，尤其在小数据集
E2	边界加权蒸馏loss	缓解砾石边缘模糊
E3	YOLOv11引导vs全图推理	检测引导在密集场景mAP↑5-8%
E4	手机端/边缘端benchmark	提出latency-power-mAP三维度指标
E5	与TinySAM/EfficientSAM对比	SOTA in geo-material domain
师弟，这就是我给你准备的弹药库！Idea 5（综合路线） 是我最推荐的，框架清晰、贡献点明确、工程上每块都有现成参考。你打算从哪个idea开始做代码验证？我直接帮你写关键模块的 PyTorch 代码！🔥 
                     
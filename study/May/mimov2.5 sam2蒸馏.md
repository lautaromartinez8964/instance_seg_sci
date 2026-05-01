明白了，核心是**密集目标实例分割**和**小目标漏检**，不是背景稀疏问题。让我针对性地重新检索。好的，我之前的理解有误。现在重新围绕**密集实例分割 + 小目标漏检**这两个核心痛点来组织。先补充检索几篇高度相关的文献。现在信息充足了。以下是我修正后的完整报告。

---

# 📚 修正版：面向密集砾石实例分割的SAM蒸馏轻量化方案

## 问题重新定义

你的核心痛点：
- ✅ **前景占比高**，图像不是稀疏场景，背景少砾石多
- ✅ **密集实例分割**：砾石紧密排列、相互粘连/重叠，需要精确的实例级分离
- ✅ **小砾石漏检**：SAM类模型的1/16 stride特征图丢失了大量细粒度空间信息，小砾石mask质量极差甚至完全漏掉
- ✅ **轻量化**：目标参数量几M~十几M，可部署在手机端

---

## 综合文献（7篇核心+2篇辅助）

### 📄 论文 1：PicoSAM2 — 极致轻量的SAM2蒸馏
**arXiv 2506.18807 | 2025.06 | Sony IMX500团队**

- **核心问题**：SAM2无法部署在传感器芯片内
- **方法创新**：Depthwise separable U-Net学生 + 从SAM2做encoder+decoder联合蒸馏 + 定点提示编码；量化后仅1.22MB
- **实验亮点**：**1.3M参数，36M MACs**；COCO 51.9% mIoU；蒸馏带来LVIS +3.5% mIoU / +5.1% mAP
- **代码**：未开源
- **对你的价值**：证明了SAM2的语义可以压缩到1.3M级别。但它的设计面向通用分割，**没有针对密集小目标做优化**。

---

### 📄 论文 2：EfficientSAM3 — 渐进式分层蒸馏
**arXiv 251.15833 | 2025.1**

- **核心问题**：SAM3统一架构（Hiera backbone + DETR检测器 + dense-memory tracker）无法端侧部署
- **方法创新**：三阶段PHD——①Encoder蒸馏（SA-1B对齐特征）②Temporal Memory蒸馏（Perceiver压缩时空特征）③端到端微调；学生backbone可选RepViT/TinyViT/EfficientViT
- **实验亮点**：VOS基准上性能-效率帕累托前沿；灵活适配不同算力预算
- **代码**：✅ https://github.com/SimonZeng7108/efficientsam3
- **对你的价值**：三阶段渐进蒸馏框架可直接复用，RepViT等backbone本身是几M参数级。

---

### 📄 论文 3：De-LightSAM — 自动prompt的轻量SAM
**arXiv 2407.14153 | Under Review**

- **核心问题**：SAM需要手动prompt、计算量大、解码器泛化差
- **方法创新**：①**轻量DC-Encoder**（域可控编码器）产生多模态判别特征 ②**SP-Generator**自动生成dense prompt embedding（**无需人工点击**）③QM-Decoder为每个模态独立解码通道 ④多模态解耦知识蒸馏MDKD
- **实验亮点**：**仅用SAM-H 2.0%的参数**，多种医学分割任务超越SOTA
- **代码**：✅ https://github.com/xq141839/De-LightSAM
- **对你的价值**：SP-Generator的"自动dense prompt"思路非常适合密集砾石——砾石数量多、无法逐个点击，必须自动生成区域提示。2%参数 ≈ 约6M，正好在你的目标范围内。

---

### 📄 论文 4：MGD-SAM2 — 多视角细节增强的高分辨率分割
**arXiv 2503.23786 | 2025.03**

- **核心问题**：SAM2在高分辨率类无关分割中丢失细粒度细节（低分辨率mask预测 + 无法直接处理高分辨率输入）
- **方法创新**：①**MPAdapter**适配SAM2编码器，增强局部细节和全局语义提取 ②**MCEM**在多尺度内聚合局部纹理 ③**HMIM**跨尺度交互全局上下文 ④**DRM**逐步恢复高分辨率mask，补偿直接上采样的细节损失
- **实验亮点**：多个高分辨率/正常分辨率数据集上超越SOTA
- **代码**：✅ https://github.com/sevenshr/MGD-SAM2
- **对你的价值**：**直接对口你的小砾石漏检问题**。DRM（细节精细化模块）可以补偿SAM2的1/16 stride导致的细粒度丢失；HMIM的多尺度交互适合处理不同粒径砾石。

---

### 📄 论文 5：SOPSeg — 遥感小物体实例分割
**arXiv 2509.03002 | 2025.09**

- **核心问题**：SAM的1/16特征分辨率导致小物体严重丢失细节；遥感领域无专用小物体实例分割数据集
- **方法创新**：①**区域自适应放大策略**（region-adaptive magnification）保留细粒度细节 ②定制decoder集成**边缘预测+渐进式精细化** ③面向遥感的**旋转框prompt机制**
- **实验亮点**：小物体分割超越现有方法；发布基于SODA-A的小物体实例分割数据集
- **代码**：将开源
- **对你的价值**：砾石本质是"密集小目标"，区域放大+边缘精细化思路直接适用。旋转框prompt对不规则砾石形状也比水平框更合适。

---

### 📄 论文 6：GDD — 面向密集预测的生成去噪蒸馏
**arXiv 2401.08332 | GitHub开源**

- **核心问题**：现有蒸馏方法对空间位置无差别对待，密集预测任务中存在大量冗余
- **方法创新**：提出"概念特征"（concept feature）→ 加入随机噪声 → 通过浅层网络生成实例特征 → 与teacher的实例特征对齐；本质是**在蒸馏中引入空间感知的去噪过程**
- **实验亮点**：PSPNet(ResNet-18) mIoU 69.85→74.67，DeepLabV3 73.20→7.69（Cityscapes）；在目标检测、实例分割、语义分割上均SOTA
- **代码**：✅ https://github.com/ZhgLiu/GDD
- **对你的价值**：**核心启发**——蒸馏时不应对所有像素/位置一视同仁，应该让student关注"有信息量"的区域。在砾石场景中，砾石之间的边界区域和小砾石区域是最需要蒸馏知识的"困难区域"。

---

### 📄 论文 7：SAM2-UNet — SAM2编码器+U-Net解码器
**arXiv 2408.08870 | Visual Intelligence 2026**

- **核心问题**：如何将SAM2的强大编码能力迁移到下游密集预测任务
- **方法创新**：直接复用SAM2的Hiera backbone作为编码器 + 经典U-Net解码器 + adapter高效微调
- **实验亮点**：在伪装目标检测、显著性检测、息肉分割等多个任务上"简单粗暴"SOTA
- **代码**：✅ https://github.com/WZH0120/SAM2-UNet
- **对你的价值**：证明了**SAM2 encoder + 轻量decoder**的范式有效性。你的轻量化方案可以借鉴这个思路——保留SAM2 encoder的部分层（或蒸馏后的轻量encoder），搭配针对密集实例分割设计的decoder。

---

### 📄 辅助论文 8：De-homogenized Queries — 密集检测去同质化
**arXiv 2502.07194 | 2025.02**

- **核心问题**：DETR中同质query导致密集场景下重复预测和漏检
- **方法创新**：可学习差异化编码（differentiated encoding），query之间通过差异化信息通信替代self-attention；联合loss同时考虑位置和置信度
- **实验亮点**：CrowdHuman 93.6% AP，39.2% MR-2，84.3% JI，超越SOTA；参数比Deformable DETR减少8%
- **代码**：未开源
- **对你的价值**：砾石密集排列的本质和人群密集检测类似。"差异化query"思想可以引入你的实例分割head——让每个砾石实例的query具有差异性，避免NMS后重复预测和小砾石被大砾石"吞噬"。

---

### 📄 辅助论文 9：CLoCKDistill — 面向DETR的蒸馏
**arXiv 2502.10683 | 2025.02**

- **核心问题**：现有KD方法盲目信任teacher，且无法蒸馏transformer的全局上下文
- **方法创新**：蒸馏transformer encoder输出（含全局上下文）+ 用目标位置信息enrich蒸馏特征 + 基于GT创建target-aware query使student和teacher关注一致的encoder memory区域
- **实验亮点**：学生检测器提升2.2%~6.4%（COCO/KITTI）
- **代码**：未开源
- **对你的价值**：如果你的方案中包含DETR-like检测头（如用query做实例分割），这篇的"位置感知蒸馏"可以直接借鉴。

---

## ⭐ 修正后的魔改 Idea（针对密集+小目标+轻量化）

### Idea 1：DenseGravel-SAM — 高分辨率特征蒸馏 + 差异化实例query

**问题分析：**
砾石密集场景有两大困难：①小砾石在1/16 stride特征图中只有几个像素甚至消失 ②粘连砾石的边界模糊，NMS后容易合并或漏检。

**技术路线：**
```
┌────────────────────────────────┐
│              SAM2-H (Teacher, ~600M)             │
│  提取多层特征: F_P2(1/4), F_P3(1/8), F_P4(1/16)  │
│  提取mask logits: M_teacher                      │
└────────────────────┬────────────────────────────┘
                     │ 多层特征蒸馏 + mask蒸馏
                     ▼
┌────────────────────────────────┐
│         DenseGravel-SAM Student (~8-12M)         │
│                                                  │
│  ┌─ Lightweight Encoder (RepViT-S/MobileNetV4)─┐│
│  │  输出: S_P2(1/4), S_P3(1/8), S_P4(1/16)    ││
│  └────────────────────────────────┘│
│                     │                            │
│  ┌─ High-Resolution Feature Enhancement ────────┐│
│  │  (借鉴MGD-SAM2的HMIM + DRM思路)              ││
│  │  • 跨尺度交互: P2↔P3↔P4 多尺度注意力融合     ││
│  │  • 细节恢复: P4→P3→P2 渐进上采样+细节补偿    ││
│  │  • 关键: 保留P2(1/4)的高分辨率信息            ││
│  └────────────────────────┘│
│                     │                            │
│  ┌─ Differentiated Instance Query Head ─────────┐│
│  │  (借鉴De-homogenized Queries的差异化编码)     ││
│  │  • 每个query编码位置+尺寸+纹理先验            ││
│  │  • query间差异化通信 (非同质self-attention)   ││
│  │  • 避免密集场景下重复预测/小砾石被吞噬        ││
│  └────────────────────────┘│
│                     │                            │
│  ┌─ Boundary-Aware Mask Head ───────────────────┐│
│  │  (借鉴SOPSeg的边缘精细化)                     ││
│  │  • 辅助边缘检测分支                           ││
│  │  • Boundary loss + Dice loss + BCE loss       ││
│  │  • 解决粘连砾石的分离问题                     ││
│  └────────────────────────┘│
└────────────────────────────────┘
```

**蒸馏Loss设计（核心创新点）：**
```python
# 1. 多尺度特征蒸馏 (Feature-level KD)
L_feat = Σ_s ||F_s - Align(F_s_teacher)||²  # s ∈ {P2, P3, P4}
# 关键: 用更高的权重蒸馏P2层特征，因为小砾石信息主要在高分辨率层

# 2. 小目标感知蒸馏权重 (Small-Object-Aware KD Weight)
# 对每个空间位置，按GT instance面积赋予不同权重
# 小砾石区域 → 高权重 (重点蒸馏)
# 大砾石区域 → 正常权重
w_spatial = 1 + α * (1 - instance_area / max_area)  
L_spatial_kd = Σ w_spatial * ||F_student - F_teacher||²

# 3. Mask蒸馏
L_mask = KL(p_student || p_teacher)  # soft mask logits

# 4. 边界辅助Loss
L_boundary = BCE(edge_pred, edge_gt)

# 5. 实例分割Loss
L_instance = L_cls + L_mask_gt + L_dice

# 总Loss
L_total = λ1*L_feat + λ2*L_spatial_kd + λ3*L_mask + λ4*L_boundary + L_instance
``

**潜在贡献点：**
1. **小目标感知蒸馏权重**：首次在SAM蒸馏中引入空间自适应权重，让student在小砾石区域获得更强的监督信号
2. **差异化实例query**：解决密集砾石场景下query同质化导致的漏检/重复
3. **高分辨率特征保留**：在轻量encoder中保留1/4 stride特征，配合渐进上采样恢复细节

**实验设计：**
| 实验 | 内容 | 指标 |
|---|---|
| 主实验 | 对比SAM2-H, MobileSAM, EdgeSAM, PicoSAM2 | mAP₅₀, mAP₅₀:₉₅, 小砾石recall |
| 消融1 | 蒸馏层级: 只蒸P4 vs P3+P4 vs P2+P3+P4 | 各粒径mAP |
| 消融2 | 小目标感知权重 on/off | 小砾石(<2cm) recall变化 |
| 消融3 | 差异化query vs 同质query | 密集区域重复预测率 |
| 消融4 | 边界loss权重敏感性 | 边界F1-score |
| 端侧部署 | ONX→TFLite/CoreML, 手机推理 | 延迟, 模型大小 |

---

### Idea 2：GravelProtoNet — 砾石原型引导的密集实例蒸馏

**问题分析：**
密集砾石的核心难题是**粘连分离**——两个紧贴的砾石在像素级别几乎无法区分。传统mask NMS在这种场景下要么合并两个instance，要么产生碎片化mask。需要引入**砾石形状先验**来辅助分离。

**技术路线：**
```
┌────────────────────────────────┐
│  Step 1: SAM2-H Teacher 生成高质量mask          │
│  (利用SAM2的强大泛化能力标注砾石)               │
└────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────┐
│  Step 2: 砾石原型学习 (Gravel Prototype Bank)    │
│                                                  │
│  K个可学习原型向量 {p_1, ..., p_K}               │
│  每个原型编码一种"砾石类型"的特征:               │
│    • 纹理 (粗糙/光滑/层理)                       │
│    • 形状 (圆形/棱角/扁平)                       │
│    • 尺寸分布 (大/中/小)                         │
│                                                  │
│  每个GT instance → 匹配最近原型 → 原型EMA更新   │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌────────────────────────────────┐
│  Step 3: 轻量Student网络 (~8-12M)               │
│                                                  │
│  ┌─ Lightweight Encoder (RepViT-S) ───────────┐│
│  └────────────────────────────────┘│
│                     │                            │
│  ┌─ Prototype-Guided Mask Decoder ────────────┐│
│  │  • 原型向量作为额外"语义prompt"              ││
│  │  • 与图像特征做cross-attention               ││
│  │  • 每个位置的特征与所有原型计算相似度        ││
│  │  → 生成K个"原型响应图" (prototype heatmap)  ││
│  │  → 合并为最终instance mask                   ││
│  │                                              ││
│  │  粘连分离机制:                               ││
│  │  相邻砾石如果匹配不同原型 → 自然分离         ││
│  │  相邻砾石匹配相同原型 → 用边缘响应分割       ││
│  └────────────────────────┘│
│                     │                            │
│  ┌─ Multi-Scale FPN Head (处理不同粒径) ───────┐│
│  │  P2(1/4): 小砾石 (<2cm)                     ││
│  │  P3(1/8): 中砾石 (2-10cm)                   ││
│  │  P4(1/16): 大砾石 (>10cm)                   ││
│  │  每个尺度独立预测mask + 融合                 ││
│  └────────────────────────────────┘│
└────────────────────────────────┘
``

**蒸馏策略：**
``
Teacher: SAM2-H → 蒸馏3个层次的知识
  ① Feature KD: encoder各层特征对齐
  ② Prototype KD: 蒸馏SAM2对砾石的隐式类别理解
     → SAM2虽然没有显式类别，但其特征空间中
       不同类型砾石已有聚类倾向
     → 用teacher特征做prototype的监督信号
  ③ Mask KD: soft mask logits对齐

特别设计: Hard Mining KD
  → 对teacher预测置信度低的区域 (通常是小砾石/粘连边界)
  → 增加蒸馏loss权重
  → 强制student在这些"困难区域"学到更多
```

**潜在贡献点：**
1. **砾石原型记忆机制**：无需显式类别标签，通过可学习原型自动区分不同类型砾石
2. **原型引导的粘连分离**：利用形状/纹理先验辅助分离紧密接触的砾石
3. **困难样本挖掘蒸馏**：对小砾石和粘连边界区域施加更强的蒸馏监督
4. **跨场景few-shot迁移**：原型bank可以快速适应新河床/海滩场景

**实验设计：**
| 实验 | 内容 |
|---|---|
| 粘连分离 | 定义"粘连砾石对"子集，单独评估分离精度 |
| 小目标recall | 按粒径分组: <1cm, 1-2cm, 2-5cm, 5-10cm, >10cm |
| 原型分析 | K值敏感性 (8/16/32/64)；原型可视化 (是否学到纹理/形状) |
| 跨场景 | A河床训练 → B海滩测试 (few-shot适应) |
| 蒸馏消融 | Feature KD / Prototype KD / Hard Mining KD 各自贡献 |

---

### Idea 3：Multi-Resolution Distillation Cascade — 多分辨率蒸馏级联

**问题分析：**
SAM2的Hiera encoder输出1/16 stride特征，这是小砾石丢失的根本原因。但直接用高分辨率特征会导致计算量爆炸。需要**在蒸馏过程中显式地让student学会在不同分辨率下工作**。

**技术路线：**
``
Phase 1: Coarse-to-Fine 蒸馏
┌────────────────────────────────┐
│  Teacher SAM2-H:                                  │
│    T_1/16: 全局语义特征 (砾石区域/背景)            │
│    T_1/8:  中等粒度特征 (砾石个体)                 │
│    T_1/4:  细粒度特征 (砾石边界/纹理)              │
│                                                   │
│  Student: 轻量encoder (RepViT-S, ~5M)             │
│    S_1/16: 蒸馏T_1/16 → 学全局砾石分布             │
│    S_1/8:  蒸馏T_1/8  → 学砾石个体区分             │
│    S_1/4:  蒸馏T_1/4  → 学边界和小砾石细节         │
│                                                   │
│  蒸馏顺序: 先粗后细                               │
│    Step 1: 只蒸馏S_1/16 (让student先理解场景)      │
│    Step 2: +蒸馏S_1/8  (学习个体分离)              │
│    Step 3: +蒸馏S_1/4  (学习细节)                  │
└────────────────────────────────┘

Phase 2: 多分辨率Cascade推理
┌────────────────────────────────┐
│  输入图像 → Student Encoder → 多尺度特征          │
│       │                                           │
│       ├─→ 1/16特征 → Coarse Head                  │
│       │    → 生成粗粒度proposal (砾石区域)         │
│       │    → 过滤明显背景区域                       │
│       │                                           │
│       ├─→ 1/8特征 → Medium Head                   │
│       │    → 在proposal区域内精细化                 │
│       │    → 分离中等大小砾石                       │
│       │                                           │
│       └─→ 1/4特征 → Fine Head                     │
│            → 在1/8输出的指导下                      │
│            → 恢复小砾石mask + 精确边界              │
│                                                   │
│  三尺度输出 → 渐进融合 → 最终实例分割结果          │
└────────────────────────────────┘
```

**关键设计细节：**
``
# 1. 渐进蒸馏的课程学习 (Curriculum Learning)
# 先让student学会"在哪有砾石"，再学"砾石边界在哪"
epoch 0~30:    L = L_feat(1/16) + L_mask_coarse
epoch 30~60:   L += L_feat(1/8) + L_mask_medium  
epoch 60~100:  L += L_feat(1/4) + L_mask_fine + L_boundary

# 2. 小砾石recall增强策略
# 在Fine Head中，对小砾石区域做"放大-分割-缩回"
small_regions = detect_coarse(1/8 features)  # 先找到小砾石大概位置
for region in small_regions:
    crop_1/4 = crop(feature_1/4, region, scale=2x)  # 放大2x
    fine_mask = FineHead(crop_1/4)  # 在放大区域做精细分割
    paste_back(fine_mask, region)  # 缩回原图

# 3. Instance-aware 蒸馏权重
# 对每个instance，按面积分配蒸馏权重
# 小instance → 5x权重
# 中instance → 2x权重  
# 大instance → 1x权重
``

**潜在贡献点：**
1. **渐进式多分辨率蒸馏**：先粗后细的课程学习策略，比一次性蒸馏所有层更稳定
2. **小砾石放大-分割-缩回机制**：在推理时对小区域做局部放大处理，显著提升小目标recall
3. **轻量多尺度cascade head**：三个分辨率各配一个轻量head，只在需要的区域做精细计算

**实验设计：**
| 实验 | 内容 | 关键指标 |
|---|---|
| 渐进蒸馏 vs 一次性蒸馏 | 对比两种蒸馏策略 | 小砾石recall, mAP |
| 放大机制消融 | 有/无 2x放大-缩回 | 小砾石(<2cm) recall |
| 分辨率贡献 | 依次去掉1/4, 1/8, 1/16 | 各粒径精度 |
| 计算效率 | cascade推理 vs 全分辨率推理 | FLOPs, 延迟 |

---

### Idea 4：SAM2 Encoder Distillation + Density-Aware NMS-Free Head

**问题分析：**
传统NMS在密集砾石场景中是灾难性的——IoU阈值设高了小砾石被合并，设低了大砾石产生重复检测。需要**完全抛弃NMS**，用端到端的方式做密集实例分割。

**技术路线：**
```
┌────────────────────────────────┐
│  SAM2-H Teacher: 特征 + mask logits              │
└────────────┬────────────────────┘
                     │ 蒸馏
                     ▼
┌─────────────────────────┐
│  Student Encoder (RepViT-S, ~5M)                 │
│  蒸馏loss: multi-layer feature KD               │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────┐
│  Density-Aware NMS-Free Instance Head            │
│                                                  │
│  核心: 借鉴De-homogenized Queries的差异化思想    │
│                                                  │
│  ┌─ Density Map Predictor ─────────────────┐│
│  │  预测每个位置的砾石密度                      ││
│  │  高密度区域 → 分配更多query                  ││
│  │  低密度区域 → 分配更少query                  │
│  └────────────────────────────┘│
│                                                  │
│  ┌─ Differentiated Query Generator ───────────┐│
│  │  query = position_encoding + size_prior      │
│  │         + texture_code + density_code        ││
│  │  每个query天生不同，无需NMS去重              ││
│  └────────────────────────┘
│                                                  │
│  ┌─ Hungarian Matching (端到端训练) ─────────┐│
│  │  一对一匹配: 每个GT砾石只匹配一个query       ││
│  │  天然消除重复检测                             ││
│  │  背景query → 学会不输出                       ││
│  └────────────────────────────────┘
│                                                  │
│  ┌─ Small-Object Recall Enhancement ──────────┐│
│  │  在Hungarian匹配中，对小砾石GT使用            ││
│  │  "匹配优先级提升"策略:                        ││
│  │  cost_small = cost * 0.5 (降低匹配代价)      ││
│  │  → 小砾石更容易匹配到query，减少漏检         ││
│  └────────────────────────────────┘
└─────────────────────────┘
```

**蒸馏中的密度感知设计：**
```python
# Density-Aware Feature Distillation
# 核心: 在高密度区域蒸馏更精细的特征

def density_aware_kd_loss(feat_s, feat_t, density_map):
    """
    density_map: 预测的砾石密度图 (每像素的砾石数量估计)
    高密度区域 → 更高蒸馏权重
    """
    weight = 1.0 + beta * density_map  # beta=2.0
    loss = weight * (feat_s - feat_t.detach()) ** 2
    return loss.mean()

# 小砾石recall辅助loss
# 在训练中，对面积 < threshold 的instance施加额外的分类loss
# 防止小砾石的query被训练为"背景"
def small_object_recall_loss(cls_pred, cls_gt, instance_area, area_threshold=100):
    small_mask = instance_area < area_threshold
    if small_mask.any():
        loss_small = F.cross_entropy(cls_pred[small_mask], cls_gt[small_mask])
        return loss_small * 3.0  # 放大3倍权重
    return 0
```

**潜在贡献点：**
1. **密度自适应query分配**：根据砾石密度动态调整query数量，密集区域获得更多query
2. **NMS-free密集实例分割**：完全消除NMS，避免密集场景下的漏检/重复
3. **小砾石匹配优先级提升**：在Hungarian匹配中显式保护小砾石不被忽略
4. **密度感知蒸馏权重**：高密度区域获得更强的蒸馏监督

**实验设计：**
| 实验 | 内容 |
|---|---|
| NMS vs NMS-free | 对比传统NMS和端到端方案在密集场景的表现 |
| 密度自适应消融 | 固定query数 vs 密度自适应分配 |
| 小砾石保护 | 有/无匹配优先级提升 → 小砾石recall |
| 蒸馏权重 | 均匀权重 vs 密度感知权重 |

---

## 📋 四个Idea对比总结

| 维度 | Idea 1: DenseGravel-SAM | Idea 2: GravelProtoNet | Idea 3: Multi-Res Cascade | Idea 4: NMS-Free Head |
|---|---|
| **核心创新** | 小目标感知蒸馏 + 差异化query | 砾石原型引导粘连分离 | 渐进多分辨率蒸馏 + 放大机制 | 密度自适应NMS-free |
| **解决痛点** | 小砾石漏检 + 密集重复 | 粘连砾石分离 | 小砾石recall提升 | 密集场景NMS灾难 |
| **新颖性** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **可行性** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **参数量** | 8-12M | 8-15M | 8-12M | 6-10M |
| **投稿故事** | 稳扎稳打，易讲 | 有亮点，故事新颖 | 工程价值高 | 理论贡献强 |

## 🎯 我的最终建议

**最优组合：Idea 1的框架 + Idea 4的NMS-free head + Idea 3的小砾石放大机制**

具体来说：
1. **Encoder**：RepViT-S/MobileNetV4 轻量backbone，从SAM2-H做多层特征蒸馏（Idea 1/3）
2. **Decoder**：密度自适应NMS-free实例head（Idea 4），用差异化query避免密集场景重复检测
3. **小目标增强**：在推理时对小砾石区域做2x局部放大-分割-缩回（Idea 3）
4. **蒸馏Loss**：小目标感知的空间自适应蒸馏权重（Idea 1）+ Hard Mining（Idea 2）

这个组合同时解决了你的两个核心痛点，且参数量可控在10M以内。

需要我对某个idea继续深入展开，或者帮你进一步检索补充文献吗？
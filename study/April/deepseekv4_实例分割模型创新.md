# 总工程师蓝图：面向端侧密集砾石分割的几何驱动与知识蒸馏框架

## 0. 总体愿景与核心叙事

我们正在构建一套**轻量级、几何感知、端侧可部署**的密集砾石实例分割系统。它的核心叙事是：

> **现有轻量分割器在密集砾石场景中失效，根本原因在于它们是“外观驱动的像素分类器”，缺乏对砾石几何结构（轮廓、边界、缝隙）的理解。我们通过三个互补的创新模块，将一个普通的一阶段检测器改造为“几何驱动的形状理解器”：**
> 1. **DT-FPN**：在 neck 中显式注入距离变换几何先验，让网络学会区分中心与边界。
> 2. **NWD Assigner**：用小目标友好的 Wasserstein 距离进行正负样本匹配，解决密集小目标的分配不稳定性。
> 3. **SAM2 不确定性几何蒸馏（UAGD）**：从大模型中萃取精细的几何结构知识，以零推理开销注入轻量学生网络。
>
> 最终，我们得到一个在手机端实时运行、却能像 SAM 一样理解砾石几何的分割器。

---

## 1. 基础模型与先决条件

**基座**：RTMDet-Ins tiny （mmdetection 实现）  
**Backbone 候选**：CSPNeXt tiny / 轻量 VMamba hybrid（保留前两阶段卷积，后两阶段替换为 VMamba-2241，作为结构增强分支，不计入核心贡献）  
**已具备条件**：2000 张自建砾石数据集（全标注多边形），训练/验证/测试 1400/300/300，图像分辨率不定（640×640 为主，部分 320×320 或更大）。

---

## 2. 模块一：Shared-DT DT-FPN（地理先验注入）

### 2.1 目的
为标准 PAFPN 的 top-down 融合提供**显式几何门控**：在实例中心区域增强高层语义，在边界区域保留浅层细节，从而改善密集邻接实例的特征分离。

### 2.2 结构细节
- **位置**：替换 RTMDet-Ins 原有的 `CSPNeXtPAFPN` 为 `DTCSPNeXtPAFPN`。
- **模式**：`dt_mode='shared'`（shared 全局 DT 图 ，不加每层独立refine）。
 
  - **当前问题**：原先从 raw backbone inputs (C3/C4/C5) 解码，语义不对齐 → 产生模糊 DT。
  - **修改后**：从 top-down 融合后的 **inner features**（如 P3_lateral, P4_lateral, P5_lateral）解码。这些特征已进行跨层语义对齐，能直接产生更清晰的 DT 图。
  - **具体实现**：在 `DTCSPNeXtPAFPN` 的 `forward` 中，将 `self._decode_shared_dt_map(backbone_outs)` 改为 `self._decode_shared_dt_map([P3_lateral, P4_lateral, P5_lateral])`。
- **DT 头容量**：`dt_head_channels=128`（对齐 Mask R-CNN 版），`guided_levels=[0,1,2]`（引导三层）。
- **训练损失**：共享 DT 图有显式 L2 loss（与 GT 的 DT 图），`loss_neck_dt` 权重 `1.0`。
- **数据增强**：**关键修复** —— 不含任何 Mosaic 或 MixUp，仅用随机缩放、裁剪、翻转、亮度抖动。`switch_epoch=0`，全程保持纯净几何监督。

### 2.3 代码修改清单（给 GPT Agent）
1. 创建 `DTCSPNeXtPAFPN`（或修改已有）：
   - `_decode_shared_dt_map` 使用 `inner_laterals`（top-down 融合后的特征列表）作为输入。
   - 实现 `shared` 模式：全局共享 DT 图 。
   - 输出 refined DT 列表，用于门控融合。
2. 配置文件（`rtmdet_ins_tiny_dt_fpn_refined_120e.py`）：
   - `neck=dict(type='DTCSPNeXtPAFPN', dt_mode='shared_refined', dt_head_channels=128, guided_levels=[0,1,2])`
   - `loss_neck_dt=dict(type='MSELoss', loss_weight=1.0)`
   - 数据增强移除 `CachedMosaic`, `CachedMixUp`，`switch_epoch=0`。

---

## 3. 模块二：NWD(Normalized Wasserstein Distance)-based Dynamic Assigner（小目标友好匹配）

### 3.1 问题定位
RTMDet-Ins 默认用 IoU 进行动态标签分配，小目标对微小偏移极为敏感，导致正负样本分配不稳定，尤其密集场景中的小砾石容易漏赋正样本。

### 3.2 方案：NWD 匹配 + IoU 质量回填
我们不直接用 NWD 替代 IoU 作为质量分数（那会导致大目标 AP 崩溃），而是执行**双度量分离**：
- **匹配阶段**：用 NWD 计算 cost，决定每个 anchor/point 匹配到哪个 GT。
- **质量标签生成**：匹配完成后，对分配到的正样本，用原始 **IoU** 重新计算 `max_overlap`，用作 QualityFocalLoss 的目标和回归损失权重。

### 3.3 具体实现
- 在 mmdet 的 `DynamicSoftLabelAssigner` 中，改写 `assign` 方法：
  1. 用 NWD calculator 计算 `pairwise_overlaps`（作为 cost）。
  2. 执行原有的 `dynamic_k` 和 `soft_label` 逻辑，得到 `assigned_gt_inds`。
  3. 对 `assigned_gt_inds > 0` 的样本，用原始 `BboxOverlaps2D`（IoU）重新计算他们与对应 GT 的 IoU，覆盖 `max_overlaps`。
- 代码片段思路：
  ```python
  # 原生的 pairwise_iou 保留一份
  iou_overlaps = self.iou_calculator(bboxes, gt_bboxes)
  # 用 NWD 做匹配
  nwd_overlaps = self.nwd_calculator(bboxes, gt_bboxes)
  # ... 用 nwd_overlaps 完成 assign
  # 回填真实 IoU
  pos_inds = assigned_gt_inds > 0
  if pos_inds.any():
      assigned_gt_inds_pos = assigned_gt_inds[pos_inds] - 1
      assign_result.max_overlaps[pos_inds] = iou_overlaps[pos_inds, assigned_gt_inds_pos]
  ```
- **常数 C**：使用已有的 8.0，保持不变。

### 3.4 验证指标
- 应观察到 `mAP_s` 提升，`mAP_l` 不再崩塌，总体 mAP 提升 +0.5~1.0。

---

## 4. 模块三（主创新）：不确定性感知的SAM2几何蒸馏 (UAGD v2)

### 4.1 三级蒸馏架构

| 级别 | 名称 | 蒸馏内容 | 学生侧接口 | 优先级 |
|------|------|----------|------------|--------|
| L1 | 响应蒸馏 | **边界置信度图 $B_{tea}$（主, λ=1.0）** + 距离变换图 $D_{tea}$（辅, λ=0.2） | Shared-DT Decoder + 轻量边界头 | ✅ 必须 |
| L2 | 特征蒸馏 | SAM2图像编码器降维特征（PCA→64维） | Adapter对齐模块 | ⚠️ 独立消融实验 |
| L3 | 关系蒸馏 | 等效半径归一化 DT 关系矩阵 | FPN P3投影头 | ✅ 必须 |

> **⚙️ 与 DT-FPN 的功能分工（重要）**：DT-FPN 已用 GT 距离变换图做特征融合门控，若 L1 再以同等权重蒸馏 SAM2 的 $D_{tea}$，两者功能重叠，边际收益递减。调整后，**L1 的主要增量来自 $B_{tea}$**：这是 SAM2 多次推理的边界频率统计图，质量远超人工 GT 的二值边界，是 DT-FPN 无法提供的新信息。$D_{tea}$ 保留为弱权重辅助，仅做几何结构校准。

### 4.2 教师知识离线生成（修订核心）

#### 4.2.1 提示策略：GT中心点 + 扰动（非网格提示）
对训练集每张图的**每个GT实例**：
- 取其掩码中心点 $(x_c, y_c)$
- 施加 $K=5$ 次随机扰动（$\pm 5px$ 范围内均匀采样）
- 每次运行SAM2，获得3个候选掩码及置信度分数 $q_{1,2,3}^k$

**优势**：
- 每个实例获得 $5 \times 3 = 15$ 个掩码假设
- 小砾石保证有精确的教师掩码（网格提示几乎不会命中微小目标）
- 不确定性评估是 **instance-aware** 的

#### 4.2.2 生成数据（每图三份.npy文件）

**文件1：教师距离变换图 $D_{tea}$**
$$D_{tea} = \text{DT}\left(\text{median}_{k,j}\{M^{k,j}\}\right)$$

**文件2：教师边界置信度图 $B_{tea}$**
$$B_{tea}(p) = \frac{1}{K}\sum_{k=1}^{K} \mathbb{1}[p \in \partial M^{k,1}]$$
（对每次提示的top-1掩码取边界，求平均频率）

**文件3：不确定性权重图 $W_{unc}$**（双重版本）
- 像素级：
$$W_{unc}^{pixel}(p) = 1 - \text{Var}_{k}\{M^{k,1}(p)\}$$
- 实例级：
$$W_{unc}^{inst}(i) = 1 - \frac{1}{|I_i|}\sum_{p \in I_i}\text{Var}_k\{M^{k}(p)\}$$
其中 $I_i$ 是GT实例 $i$ 的像素集合。

两种权重最终乘在一起使用：
$$W_{unc}(p) = W_{unc}^{pixel}(p) \cdot W_{unc}^{inst}(i_p) \cdot (1 - \text{mean}_k(\text{std}_j(q^{k,j})))$$

> **💡 可选扩展：密度感知权重 $W_{dense}$**（来自综合讨论，作为消融实验行 I）
> 
> 在 $W_{unc}$ 基础上叠加局部砾石密度感知权重，在高密度接缝集中区域提高蒸馏损失权重：
> $$W_{dense}(p) = 1 + \gamma \cdot \rho(p), \quad \gamma=0.5$$
> 其中 $\rho(p)$ 为以像素 $p$ 为中心、半径 $r=32\text{px}$ 的邻域内 GT 掩码覆盖率（归一化至 $[0,1]$）。
> 最终组合权重：$W_{total}(p) = W_{unc}(p) \cdot W_{dense}(p)$
> 
> **实现**：5 行代码可实现，通过配置 `use_density_weight=True` 开关控制，**不计入主线方案，仅作第 9 条消融验证**。

#### 4.2.3 文件4（备选）：教师特征图 $F_{tea}$
**存储计算重算**：
- SAM2-L图像编码器输出：1024维，1/16空间分辨率
- 640×640输入 → 40×40空间位置
- 单张图：$40 \times 40 \times 1024 \times 4 \text{ bytes} = 6.6\text{MB}$
- PCA压缩到64维：$40 \times 40 \times 64 \times 4 = 0.41\text{MB}$/张
- 1400张训练图总计：$\approx 570\text{MB}$

**结论**：存储完全可控，Level 2不应因此被砍。

#### 4.2.4 离线运行时间重算
- SAM2-L在RTX 5090上单次推理约80-120ms (FP16, 640×640)
- 每张图：N个实例 × 5次提示 × 0.1s
- 假设平均每图50个实例：$50 \times 5 \times 0.1 = 25s$/张
- 1400张训练图总时间：$1400 \times 25s = 35000s \approx 9.7h$
- **实际约8-12小时**（取决于实例数量和系统IO）

**结论**：一晚上跑完，后续训练多次复用，完全可以接受。

### 4.3 学生端新增模块（训练专用，推理删除）

#### 4.3.1 几何解码头（L1响应蒸馏用）
接在FPN P3输出上，约0.08M参数：
```
FPN P3 (C=128) -> 3×3 DWConv + GN + ReLU -> 3×3 DWConv + GN + ReLU -> 1×1 Conv -> (D_stu, B_stu)
```
- `B_stu`：1通道，预测边界置信度图（Sigmoid激活）——**主要输出**，对应 $\lambda_b=1.0$ 的强监督信号
- `D_stu`：1通道，预测DT图——辅助输出，对应 $\lambda_d=0.2$ 的弱监督信号

#### 4.3.2 特征对齐Adapter（L2特征蒸馏用）
```
FPN P3 (C=128) -> Linear(128→256) -> LN -> ReLU -> Linear(256→64) -> F_stu_aligned
```
- 输出64维，与PCA压缩后的教师特征空间对齐
- 使用独立学习率（主网络1/10）
- 训练时监控cosine相似度曲线，确保不单调上升（防止特征偏移）
- 梯度裁剪 `max_norm=1.0`

#### 4.3.3 关系投影头（L3关系蒸馏用）
```
FPN P3 (C=128) -> 1×1 Conv -> 128维 -> F_stu_proj（用于计算学生关系矩阵）
```

### 4.4 蒸馏损失函数

#### L1 响应蒸馏
$$\mathcal{L}_{resp} = \underbrace{\lambda_b \cdot \frac{1}{|\Omega|}\sum_{p} W_{unc}(p) \cdot \text{BCE}(B_{stu}(p), B_{tea}(p))}_{\text{主项：边界置信度蒸馏}} + \underbrace{\lambda_d \cdot \frac{1}{|\Omega|}\sum_{p} W_{unc}(p) \cdot (D_{stu}(p) - D_{tea}(p))^2}_{\text{辅项：DT结构校准}}$$
- **$\lambda_b=1.0$（主）, $\lambda_d=0.2$（辅）**
- **调权理由**：DT-FPN 已用 GT DT 图做特征门控，$D_{tea}$ 的蒸馏边际收益有限；$B_{tea}$ 来自 SAM2 多次推理的边界频率，是 DT-FPN 无法提供的精细边界先验，应作为 L1 的核心监督。
- **消融对照**：可在配置中切换 $\lambda_d=1.0, \lambda_b=0.5$（原版）vs $\lambda_d=0.2, \lambda_b=1.0$（修订版）对比 AP_boundary 增益，详见消融矩阵行 J。

#### L2 特征蒸馏（独立消融实验）
$$\mathcal{L}_{feat} = \lambda_f \cdot \mathbb{1}[W_{unc} > \tau] \cdot (1 - \text{cos}(F_{stu}^{aligned}, F_{tea}))$$
- $\lambda_f=0.3, \tau=0.5$

#### L3 关系蒸馏（核心新颖点）
**Anchor采样策略**（修订）：
- 每个batch随机选1个GT实例
- 在该实例内部采样 $N=64$ 个anchor像素
- **50%** 来自 $D_{norm} < 0.15$ 的区域（边界带）
- **50%** 来自 $D_{norm} > 0.70$ 的区域（中心带）

**DT值归一化**（Sonnet新增，解决尺度不平衡）：
对每个GT实例，用等效半径归一化：
$$D_{norm}(p) = \frac{D_{tea}(p)}{r_i}, \quad r_i = \sqrt{\frac{\text{Area}_i}{\pi}}$$
归一化后所有实例的DT值范围均为 $[0, 1]$。

**教师关系矩阵**：
$$R_{tea}(i,j) = \exp\left(-\frac{|D_{norm}(p_i) - D_{norm}(p_j)|^2}{2\sigma^2}\right), \quad \sigma=0.15$$
（$\sigma$ 基于归一化值 [0,1] 范围设定，与实例尺度无关）

**学生关系矩阵**：
$$R_{stu}(i,j) = \text{cosine\_similarity}(F_{stu}^{proj}(p_i), F_{stu}^{proj}(p_j))$$

**关系损失**：
$$\mathcal{L}_{rel} = \frac{1}{N^2}\sum_{i,j} W_{unc}^{avg}(i,j) \cdot (R_{stu}(i,j) - R_{tea}(i,j))^2$$
其中 $W_{unc}^{avg}(i,j) = \sqrt{W_{unc}(p_i) \cdot W_{unc}(p_j)}$。

### 4.5 训练策略：渐进式课程蒸馏

**Phase 1 (0% - 60% iterations)**：
- 损失 = $\mathcal{L}_{det} + \mathcal{L}_{mask} + \mathcal{L}_{dt\_sup}$
- 仅用人工GT，学生先建立基本分割和几何感知能力

**Phase 2 (60% - 100% iterations)**：
- 损失 = $\mathcal{L}_{det} + \mathcal{L}_{mask} + \mathcal{L}_{dt\_sup} + \mathcal{L}_{resp} + \alpha(t)\mathcal{L}_{rel} + [\mathcal{L}_{feat}]$
- **warm-up调度**：
  - 60%-80% iterations：$\alpha(t)$ 从0线性升到1
  - 80%-100% iterations：$\alpha(t)$ 保持1
- 学习率降为Phase 1的0.1倍
- 优化器状态从Phase 1继承

### 4.6 代码实现清单（给GPT Agent）

1. **`LoadSAMGeometricSupervision`** 新增transform：
   - 输入：`gt_instances`（每个GT实例的中心点和面积）
   - 输出：加载的 `d_tea`, `b_tea`, `w_unc`, `f_tea` (64维PCA版), `norm_radius`（每个实例的等效半径列表，用于L3的DT归一化）

2. **`GeometricDistillHead`** 新增module：
   - `dt_conv`：输出DT图
   - `bound_conv`：输出边界置信度图
   - `feature_adapter`：输出对齐特征（64维，独立消融）
   - `relation_proj`：输出关系投影特征（128维）

3. **`RelationDistillLoss`** 新增loss类：
   - 输入：`d_norm` (归一化后的DT值), `f_proj`, `w_unc`, `gt_instance_mask`
   - 内部执行：anchor采样（50%边界/50%中心）→ 构造$R_{tea}$和$R_{stu}$ → 计算MSE
   - 每batch只对1个实例计算

4. **`RTMDetInsDistill`** 模型包装器：
   - 继承自原RTMDet
   - `train_step` 中计算蒸馏loss并加入总loss
   - Phase切换逻辑基于 `self.iter` 和 `phase_switch_iter`

5. **配置文件 `rtmdet_ins_tiny_uagd_120e.py`**：
   ```python
   distill_cfg = dict(
       phase_switch_epoch=72,          # 60% of 120 epochs
       phase2_lr_factor=0.2,           # Phase 2 学习率倍数（原 0.1 → 调整为 0.2）
       loss_resp_weight=dict(
           dt=0.2,                     # 辅：与 DT-FPN 共享几何监督，弱权重
           bound=1.0,                  # 主：SAM2 边界频率图， DT-FPN 无法提供的增量信息
       ),
       loss_feat_weight=0.3,           # L2 独立消融
       loss_rel_weight=0.3,
       rel_warmup_epochs=24,           # 60-84 epochs 线性升温
       rel_sigma=0.15,
       rel_num_anchors=64,
       feat_pca_dim=64,
       use_density_weight=False,       # W_dense 开关（消融行 I 单独开启）
       density_gamma=0.5,              # W_dense 强度系数
   )
   ```


## 5. 消融实验矩阵（修订版）

| 编号 | DT-FPN | NWD (fixed) | L1 响应 | L2 特征 | L3 关系 | 不确定性加权 | 说明 |
|------|--------|-------------|---------|---------|---------|-------------|------|
| A | | | | | | | 原版RTMDet-Ins tiny |
| B | ✔ | | | | | | +几何neck |
| C | ✔ | ✔ | | | | | +小目标匹配 |
| D | ✔ | ✔ | ✔ | | | ✔ | +L1蒸馏 |
| E | ✔ | ✔ | ✔ | | ✔ | ✔ | +L1+L3蒸馏 |
| F | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | +完整三级蒸馏 |
| G | ✔ | ✔ | ✔ | | ✔ | 去掉 | 验证不确定性加权 |
| H | ✔ | ✔ | ✔ | | ✔ (无归一化) | ✔ | 验证DT归一化 |
| I | ✔ | ✔ | ✔ | | ✔ | ✔ + $W_{dense}$ | 验证密度感知加权 |
| J | ✔ | ✔ | ✔（$\lambda_d=1.0, \lambda_b=0.5$）| | ✔ | ✔ | 验证 L1 主信号切换（$B_{tea}$主 vs $D_{tea}$主） |

**核心待验证假设**：
1. L3关系蒸馏 ≥ +0.8 AP_boundary（相比只做L1）
2. 不确定性加权带来额外 +0.3-0.5 AP（在高不确定性区域更为明显）
3. DT归一化使小砾石的关系蒸馏信号不再被大砾石淩没
4. L2特征蒸馏独立贡献待测（不L1+L3耦合验证）
5. $W_{dense}$ 密度感知加权在高密度砾石区域带来额外精度提升（消融行 I）
6. **L1 主信号切换验证**：$B_{tea}$ 主导 ($\lambda_d=0.2, \lambda_b=1.0$) 相比 $D_{tea}$ 主导 ($\lambda_d=1.0, \lambda_b=0.5$)，在 AP_boundary 上的增益（消融行 J）


## 6. 实施路线图

| 阶段 | 任务 | 预计耗时 |
|------|------|----------|
| ~~**Week 1**~~ | ~~DT-FPN修复：inner features输入 + refined模式 + 去除Mosaic/MixUp~~ | ✅ **已完成**（baseline 0.272 → **0.281**，+0.009） |
| **Week 2（当前）** | NWD Assigner：匹配/质量回填 + 联合DT-FPN训练验证 | 3-4天 |
| **Week 2-3** | SAM2离线生成脚本：GT中心点提示 + 多次推理 + PCA压缩 + .npy存储 | 4-5天 (含8-12h运行) |
| **Week 3-5** | UAGD完整实现：几何头 + adapter + 关系蒸馏 + 两阶段训练 | 10-14天 |
| **Week 5-6** | 消融实验：完整矩阵训练 + 指标采集 + 可视化 | 7-10天 |
| **Week 6-7** | 对比实验 + 论文写作 | 同步 |


## 7. 风险与对策（更新）

| 风险 | 概率 | 对策 |
|------|------|------|
| GT中心点提示下SAM2单实例推理耗时超预期 | 中 | 对小实例(<100px²)合并为batch推理, 减少前向次数 |
| 关系蒸馏cost过大(64²矩阵/batch) | 低 | 降低N至32, 或每两个batch计算一次 |
| L2特征蒸馏造成检测特征偏移 | 中 | 严格独立消融, 不与L1+L3耦合; 监控cosine相似度 |
| Phase 2学习率降低过多导致收敛极慢 | 低 | Phase 2 lr设为Phase 1的0.2-0.3倍而非0.1倍 |


## 8. 论文核心贡献陈述（可直接入摘要）

1. **We propose an Uncertainty-Aware Geometric Distillation (UAGD) framework** that transfers fine-grained structural knowledge—**boundary confidence maps (primary)**, distance transform fields (auxiliary), and geometric relation graphs—from SAM2 to a lightweight on-device segmenter, achieving zero inference overhead. Critically, the boundary confidence maps derived from SAM2’s multi-hypothesis aggregation provide superior boundary precision over binary GT annotations, representing information orthogonal to and complementary with the DT supervision already captured by DT-FPN.

2. **We introduce an instance-aware multi-hypothesis uncertainty quantification mechanism**, which leverages SAM2's native multi-mask output under GT-center-point perturbations to evaluate per-instance teacher confidence, enabling selective distillation that suppresses unreliable knowledge transfer at densely cluttered boundaries.

3. **We design a geometry relation distillation paradigm with scale-normalized distance fields**, where pairwise pixel affinities in the student feature space are constrained to mirror the normalized geometric contour relationships of the teacher, ensuring equal treatment of gravels across extreme size variations.

4. **We further integrate two complementary structural enhancements**—a shared-refined DT-guided feature pyramid and a Wasserstein-based dynamic assignment strategy—that respectively improve boundary-aware feature fusion and small-instance training stability, together forming a geometry-driven lightweight segmentation framework purpose-built for dense granular scenes.

---

这份蓝图已将所有修正、改进和新增内容整合完毕。核心贡献点没有变，但方案细节的严谨性在几轮讨论后已大幅提升。你的GPT Agent现在应该可以直接基于这份蓝图开始写代码了。需要调整任何具体参数或实现细节，随时找我。
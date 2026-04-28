 # RTMDet-Ins 轻量化密集砾石实例分割整体 Idea 手册

## 0. 写作立场与总体自洽结论

在当前阶段，本文最稳妥、最自洽的主线不应继续围绕重型两阶段模型展开，也不宜把尚未完成闭环验证的训练技巧写成已被严格证实的主贡献。结合当前实现状态、已有验证结果以及潜在审稿风险，建议采用如下写作立场：

1. 主文核心方法聚焦于基于 RTMDet-Ins 的两项已落地改造：
   - Shared-DT DT-FPN
   - NWD-based Dynamic Assignment
2. VMamba 主干替换保留为候选结构创新分支，其中优先考虑更轻的 2241 设定，以及“仅将后两个 stage 替换为 VMamba”的 hybrid 路线；该分支当前适合作为结构探索与补充实验，而不宜在主文中写成已经完成闭环验证的核心贡献。
3. SAM2 蒸馏保留为扩展训练策略，作为完整框架中的潜在增强项、补充实验项或 supplementary/future work，而不作为当前主文中已经被充分验证的核心结论。
4. 因此，整篇论文的方法叙事应从“轻量化一阶段实例分割器如何通过几何先验、分配先验与可选的轻量结构替换适配密集砾石场景”展开，而不是从“我们拼接了多个模块”展开。

这个立场的好处有三点：

1. 与当前代码和实验事实一致，不触碰学术表述红线。
2. 能直接回应审稿人对 DT-FPN 和 NWD 新颖性的质疑，把贡献点落在“场景特异问题建模”和“低侵入、可迁移、轻量化设计”上。
3. 为后续若接入更有新意的 SAM2 蒸馏机制预留空间，而不损害当前稿件的完整性。

基于这个前提，本文可以将整体方法命名为：

**GDR-RTMDet-Ins: Geometry-Driven RTMDet-Ins for Lightweight Dense Gravel Instance Segmentation**

其中，Geometry-Driven 包含两个已完成的核心来源：

1. 由 distance transform 提供的显式几何先验。
2. 由 NWD 提供的小目标友好分配先验。

与此同时，本文还保留一个与主线互补的结构探索方向：

3. 由 VMamba 提供的长程依赖建模能力，但优先以轻量化 2241 或 hybrid late-stage replacement 的方式接入，而非直接使用更重的 2292 版本作为论文主卖点。

SAM2 蒸馏则被定义为：

**SAM2-assisted geometric supervision extension**

即，面向后续增强的教师监督扩展，而非当前主文结论主体。

---

## 1. 核心问题定义

本文关注的任务是密集砾石场景下的轻量化实例分割。与常规自然图像实例分割相比，该任务同时具有以下三个困难：

1. **目标尺寸小且尺度分布不均衡。** 图像中同时存在大量细小碎石和少量较大砾石，导致基于 IoU 的标签分配对微小位置扰动高度敏感。
2. **实例间粘连严重、边界模糊。** 邻近砾石之间常出现接触、重叠或弱边界现象，标准 FPN 在 top-down 融合时容易把相邻实例的特征混合，削弱实例可分性。
3. **标注边界有限且数据规模受限。** 在自建砾石数据集上，人工标注的掩膜边界不可避免地存在粗糙和不稳定现象，轻量模型更容易受到监督噪声影响。

因此，本文的目标并不是单纯追求更高精度，而是在**轻量参数量和实时友好推理开销**约束下，提升模型对密集小目标、接触边界和粗标注监督的适应能力。

形式化地，给定输入图像 $I \in \mathbb{R}^{H \times W \times 3}$，模型需要预测实例集合

$$
\mathcal{Y}=\{(b_i, m_i, c_i)\}_{i=1}^{N},
$$

其中 $b_i$、$m_i$ 和 $c_i$ 分别表示第 $i$ 个砾石实例的边界框、实例掩膜和类别。由于本文只关注单类砾石分割问题，核心挑战集中在**定位稳定性**与**边界分离能力**上，而不是类别语义区分上。

---

## 2. 方法总览

本文以 RTMDet-Ins tiny 为基础框架，在不显著增加参数量与推理复杂度的前提下，从 neck 融合与训练分配两个层面进行场景适配，形成一个面向密集砾石实例分割的轻量化一阶段框架。整体结构可概括为：

1. **轻量主干与检测头。** 采用 RTMDet-Ins 作为基础实例分割器，保留其高效 backbone、PAFPN-style neck 以及实例分割 head。
2. **Shared-DT DT-FPN。** 在 top-down 多尺度融合过程中注入显式距离变换先验，使模型在实例中心区域侧重高层语义，在相邻边界区域保留更多浅层细节。
3. **NWD-based Dynamic Assignment。** 在训练阶段使用 NWD 替代默认 IoU 作为动态软标签分配中的几何相似度度量，以提升密集小目标场景下的样本分配稳定性。
4. **Lightweight VMamba Backbone Exploration。** 作为候选结构分支，探索以 VMamba-2241 或 hybrid late-stage replacement 方式增强长程关系建模，同时维持轻量化叙事。
5. **SAM2-assisted Distillation Extension。** 作为训练增强扩展，利用高质量教师掩膜构造边界、距离变换和软掩膜监督，用于后续进一步提升边界质量和监督鲁棒性。

基于该设计，本文的主线不是重新设计一个更重的实例分割器，而是回答如下问题：

**对于密集砾石这类高度几何驱动的小目标实例分割场景，如何在轻量级 RTMDet-Ins 中显式引入几何结构约束，并保持训练和推理代价可控？**

---

## 3. 创新点一：Shared-DT DT-FPN

### 3.1 设计动机

标准 FPN 或 PAFPN 的 top-down 融合通常采用统一的逐像素加和或卷积融合方式，其默认假设是所有空间位置都应以相同方式混合深层语义与浅层细节。然而在密集砾石场景中，不同空间位置的融合需求并不一致：

1. 在实例内部区域，模型更需要稳定的高层语义支持，以抑制纹理噪声和局部遮挡。
2. 在实例间边界区域，模型更需要浅层边缘与局部结构信息，以避免相邻目标被过度语义平滑。

因此，问题的关键不在于是否“再加一个 attention”，而在于是否能构造一个**具有明确几何意义且受显式监督约束的空间引导图**，用于重写 top-down 融合规则。

### 3.2 为什么采用 shared-DT v2，而不是更复杂的 per-level 方案

本文明确采用已经在 Mask R-CNN 路线中验证过的 DT-FPN v2 形式，即：

1. 从高层到低层迭代解码出一张**共享的高分辨率 DT 图**。
2. 在每个被引导的 top-down 融合层，将这张共享 DT 图 resize 到对应分辨率。
3. 使用同一张全局几何先验图去调制不同尺度的融合过程。

该设计的核心优势在于：

1. **全局一致性。** 一张共享 DT 图在所有 top-down 路径中保持统一几何语义，避免不同层各自预测局部 DT 图带来的不一致问题。
2. **结构可解释。** 它不是“每层自己学一张注意力图”，而是“先获得一张全局几何场，再按层复用该几何场指导融合”。
3. **已在先前框架中验证。** Mask R-CNN 路线的 v2 结果表明，这种 shared-DT、从上到下迭代获取一张图再指导各层的方法优于更分散的替代形式。因此，RTMDet-Ins 中也优先采用该设置，而不再把 per-level 或 shared-refined 作为主线方案。

因此，本文在方法叙事上不应强调“我们尝试了很多 DT 变体”，而应强调：

**我们发现，对于密集实例分离，跨层共享的单一几何先验比逐层独立门控更稳定、更具可迁移性。**

### 3.3 方法描述

设 backbone 或 neck 侧向特征为 $\{F_l\}$，其中 $l$ 表示尺度层级。我们首先利用最高层特征和逐层 lateral 信息构建一个共享 DT 解码器，得到高分辨率几何先验图 $D$：

$$
D = \phi_{dt}(F_L, F_{L-1}, \dots, F_1),
$$

其中 $\phi_{dt}(\cdot)$ 表示从深到浅的共享 DT 解码过程，输出 $D \in [0,1]^{H' \times W'}$。

对于每个被引导的融合层 $l$，将共享 DT 图缩放到对应分辨率，记为 $D_l$，然后生成门控图：

$$
G_l = \sigma(\alpha_l D_l + \beta_l),
$$

其中 $\alpha_l$ 和 $\beta_l$ 为可学习参数，$\sigma(\cdot)$ 表示 Sigmoid 函数。

随后，原始 top-down 融合由统一加和改写为几何引导的插值形式：

$$
\tilde{F}_l = (1-G_l) \odot F_l^{lat} + G_l \odot U(F_{l+1}),
$$

其中 $F_l^{lat}$ 表示当前层 lateral 特征，$U(F_{l+1})$ 表示上一级特征的上采样结果。

该公式体现了本文方法与通用 attention 变体的关键差别：

1. $G_l$ 并非由当前层特征自由生成，而是由显式监督的 DT 几何场派生。
2. 所有层的门控来自同一共享几何先验，而不是相互独立的注意力权重。
3. 引导目标明确服务于实例中心与实例边界的差异化融合，而不是泛化的“增强显著区域”。

### 3.4 训练目标

为了保证共享 DT 图确实承载可解释的几何意义，本文对其施加显式距离变换监督：

$$
L_{dt} = \lambda_{dt} \cdot \text{MSE}(D, D^{gt}),
$$

其中 $D^{gt}$ 为由实例标注离线生成的距离变换图，$\lambda_{dt}$ 为损失权重。

### 3.5 学术贡献落点

针对潜在审稿意见“DT 图门控是否只是 attention 变体”，本文应当把贡献准确表述为：

**本文提出的不是一般意义上的特征重加权，而是一种由显式距离变换监督约束的跨层共享几何路由机制。该机制将实例中心到边界的连续几何结构，直接嵌入到 top-down 融合规则中，从而提升密集邻接实例的可分离性。**

换言之，本文真正的贡献点不在于“用了 DT”，而在于：

1. 选择共享单图而不是逐层局部门控。
2. 将该共享几何图直接用于重写融合公式。
3. 在轻量化 RTMDet-Ins 结构中以低侵入方式实现这一机制。

---

## 4. 创新点二：NWD-based Dynamic Assignment

### 4.1 设计动机

在密集砾石场景中，许多实例具有以下特点：

1. 边界框尺寸小。
2. 相邻实例空间间隔窄。
3. 训练初期预测框相对于真实框常存在 1 至 3 像素级微小偏移。

对于此类样本，IoU 作为分配相似度时会出现明显的不稳定性。即使预测框在语义上已经落在正确目标附近，只要中心或宽高存在轻微误差，IoU 也可能快速下降甚至趋近于零，从而造成正负样本分配抖动。

因此，本文并不试图“发明一种新的框距离”，而是针对轻量一阶段密集实例分割中的小目标分配问题，引入一个更平滑的几何相似度度量来替代默认 IoU 分配逻辑。

### 4.2 方法描述

本文在 RTMDet-Ins 的 Dynamic Soft Label Assigner 中，将 IoU calculator 替换为 NWD similarity calculator，而不改变检测头结构和框回归损失。对于任意边界框 $b=(x_1,y_1,x_2,y_2)$，我们将其建模为二维高斯分布：

$$
\mu = \left(\frac{x_1+x_2}{2}, \frac{y_1+y_2}{2}\right),
\qquad
\sigma = \left(\frac{w}{4}, \frac{h}{4}\right),
$$

其中 $w=x_2-x_1$，$h=y_2-y_1$。

对于两个候选框 $b_i$ 和 $b_j$，其 NWD 相似度定义为：

$$
S_{nwd}(b_i,b_j) = \exp\left(-\frac{\sqrt{\|\mu_i-\mu_j\|_2^2 + \|\sigma_i-\sigma_j\|_2^2}}{C}\right),
$$

其中 $C$ 为归一化常数。

该相似度具有两个优点：

1. 即使两个框尚未发生重叠，只要它们的中心和尺度接近，仍然具有连续、非零的相似度。
2. 对于小尺寸目标，NWD 相比 IoU 对微小位移更平滑，能减弱训练早期分配噪声。

### 4.3 为什么本文只改 assigner，而不改回归损失

这是一个必须主动说清的点。本文的目标是隔离并验证“小目标友好分配策略”本身的作用，因此我们只在训练样本分配阶段引入 NWD，而保持 bbox regression loss 不变。这样做有两层意义：

1. 能清晰说明性能变化来自**正负样本选择与质量估计**，而非回归损失混杂带来的额外收益。
2. 能避免与 GIoU、DIoU、CIoU 等回归损失替代项混为一谈，使本文贡献集中在 assignment 层面而不是 bounding box optimization 层面。

因此，本文建议的表述是：

**我们不将 NWD 作为新的通用回归目标，而是将其作为一种针对密集小目标场景更稳健的分配相似度，用于替换动态分配中的默认 IoU 计算。**

### 4.4 关于常数 $C=8.0$ 的合理写法

针对“固定常数缺乏理论支撑”的潜在质疑，本文不应将 $C=8.0$ 表述为理论最优值，而应写成：

1. 我们遵循 tiny-object NWD literature 的经验设置，采用统一的 $C=8.0$。
2. 在所有 RTMDet-Ins 变体中保持该常数固定，避免为单一模型额外调参。
3. 本文贡献不在于求得某个最优 $C$，而在于证明 NWD assignment 对密集砾石实例分割的适配价值。

这是一种更稳健的学术表述方式。

### 4.5 学术贡献落点

由于 NWD 本身已有文献，本文的创新点不应写成“我们提出了 NWD”。更合理的正式表述是：

**本文首次将 NWD-based similarity 以低侵入形式整合进轻量 RTMDet-Ins 的动态样本分配流程，用于缓解密集小目标实例分割中由 IoU 引起的分配不稳定问题，并与 Shared-DT 几何融合形成互补：前者稳定训练样本选择，后者改善边界表征。**

---

## 5. 候选结构分支：Lightweight VMamba Backbone Exploration

### 5.1 为什么这条线值得写进整体 idea，而不应被忽略

除了 neck 融合与 assignment 之外，密集砾石实例分割还存在一个上游结构问题：轻量 backbone 是否能够在不显著增加参数量的前提下，更有效地建立邻近实例之间的长程关系。这个问题之所以值得单独提出，是因为密集砾石场景同时具有两种相互矛盾的需求：

1. 浅层高分辨率特征需要尽量保留接缝、裂隙和边界细节。
2. 深层低分辨率特征又需要建立跨实例、跨局部纹理的全局依赖，以区分“属于同一砾石内部纹理”还是“属于相邻砾石的分界”。

VMamba 这类视觉状态空间模型在第二点上具有天然优势，但它若直接替换全部 stage，也可能在浅层高分辨率区域过度平滑细碎边界。因此，这条线最合理的研究问题不是“要不要用 VMamba”，而是：

**在轻量化约束下，VMamba 应该以多大规模、接入到哪些 stage，才能真正服务于密集砾石实例分割？**

### 5.2 为什么优先考虑 2241，而不是继续把 2292 当主线

当前仓库中已经存在 RTMDet-Ins 的 full VMamba 与 hybrid VMamba 2292 原型配置，但从论文叙事来看，2292 存在两个问题：

1. **过重。** 2292 版本更适合作为能力上界或参考对照，而不适合作为“轻量化方法”的主卖点。
2. **归因不干净。** 如果直接用更重的 2292 获得增益，审稿人很容易质疑：提升到底来自结构设计，还是单纯来自 backbone 容量增加。

因此，更自洽的路线是将 VMamba 分支收束到更轻的 2241 设定，或者至少把它表述为：

**本文优先探索 VMamba 的轻量化接入方式，而不是追求更大容量 backbone 的直接替换。**

这里的 2241 并不是一个必须夸大的“新模型名”，而是一个轻量化结构约束：相较于 2292，它通过减小后段堆叠深度，使 VMamba 更符合本文面向轻量实例分割的定位。

### 5.3 两种最合理的候选结构形式

结合当前任务特点，VMamba 分支最值得尝试的不是无约束的大改，而是以下两种形式。

#### 方案 A：Full VMamba-2241 RTMDet Backbone

该方案将 RTMDet 原有 P3/P4/P5 主干输出整体替换为轻量化 VMamba-2241 风格 backbone，用更低的 VMamba 深度设定获得长程依赖建模能力。它的优点是结构统一、叙事直接，适合作为“全量状态空间 backbone”对照组。

但它也存在明显风险：

1. 浅层高分辨率区域可能被过度平滑。
2. 小碎石边界与细缝纹理在纯 VMamba 浅层中可能不如卷积稳定。

因此，这条线更适合作为完整替换对照，而不是默认主线。

#### 方案 B：Hybrid VMamba-2241 Backbone（优先推荐）

该方案保留 CSPNeXt 的前两阶段，仅将后两个 stage 替换为轻量 VMamba。其动机非常明确：

1. **前两阶段保留卷积。** 让模型在高分辨率浅层继续利用卷积的局部边缘保持能力，尽可能减少接缝与细粒边界信息损失。
2. **后两阶段引入 VMamba。** 让模型在更低分辨率、更大感受野的区域建立跨实例上下文关系，弥补纯卷积对远程依赖建模不足的问题。

从物理直觉上看，这是一种“前局部、后全局”的分工：

1. 浅层卷积负责看清细缝和边界。
2. 深层 VMamba 负责判断跨区域的结构一致性与全局实例关系。

这条思路与当前仓库中的 hybrid backbone 原型是一致的，只是论文与后续实验中更适合把 2292 收缩为 2241 版本，以降低参数与计算负担。

### 5.4 这条分支在论文里应该怎样写才自洽

VMamba 分支当前最适合的定位，不是“本文已经证实的第三主创新”，而是：

1. 作为主体框架的候选结构增强分支。
2. 作为对 backbone-level geometry/context trade-off 的进一步探索。
3. 作为后续补充实验或二级主表中的结构消融项。

换言之，当前主文不应写成“我们提出并验证了 VMamba-2241 混合主干”，除非它已经完成了独立实验闭环。更稳妥的表述是：

**我们进一步提出一种轻量化 VMamba 接入思路，通过仅替换 RTMDet 主干后两个 stage，在保留浅层局部边界建模能力的同时，引入深层长程依赖建模能力。该结构分支作为候选增强方向，将在后续实验中系统评估。**

### 5.5 当前最推荐的实验优先级

如果后续要把 VMamba 分支补进完整实验，建议优先级如下：

1. Hybrid VMamba-2241
2. Full VMamba-2241
3. Hybrid VMamba-2292
4. Full VMamba-2292

这个顺序的核心原则是：先验证“轻量结构替换是否有效”，再看“更大容量是否进一步有效”。

### 5.6 这一分支的学术贡献落点

如果后续实验成立，这条线最值得强调的不是“用了 VMamba”，而是：

**本文探索了一种面向密集实例分割的 stage-wise heterogeneous backbone design，其中浅层保留卷积以维护高频边界信息，深层引入轻量 VMamba 以增强长程结构建模，从而在轻量化约束下实现局部几何与全局上下文的分工协同。**

---

## 6. 创新点三：SAM2-assisted Distillation Extension

### 5.1 本节在主文中的正确定位

这一部分必须谨慎表述。结合当前代码状态与审稿风险，SAM2 蒸馏不应在主文中被写成“已经被充分验证的核心贡献”。更稳妥的定位是：

1. 作为本文完整框架的扩展训练策略。
2. 作为补充实验或后续增强项。
3. 作为对高质量几何监督来源的进一步探索。

换句话说，SAM2 蒸馏在当前稿件中可以写，但应写成**扩展思路与方法设计**，而不是主结论支柱。

### 5.2 为什么仍然值得保留这条线

尽管“foundation model distillation”已经较为拥挤，但砾石场景中这条线仍有现实价值，原因在于：

1. 人工实例边界本身存在不稳定和粗糙现象，而这恰恰是密集实例分割最敏感的部分。
2. 本文已有 Shared-DT DT-FPN，天然需要更高质量的几何监督源。
3. 若后续设计得当，SAM2 并不是提供泛化语义知识，而是提供更精细的几何边界、距离场与软掩膜监督，这与本文主线是统一的。

因此，这一部分最合适的定义不是 generic SAM distillation，而是：

**SAM2-assisted geometric supervision for lightweight dense instance separation**

### 5.3 推荐的保守写法

若当前稿件以稳定过审为优先目标，建议将本节写成如下形式：

1. 我们进一步设计了一个 SAM2 辅助几何监督扩展，用于从高质量教师掩膜中构造边界图、距离变换图和软掩膜监督。
2. 该扩展仅在训练阶段使用，推理阶段不引入额外计算。
3. 由于当前工作重点在于验证 Shared-DT DT-FPN 与 NWD assignment 的主体有效性，SAM2 扩展将在后续版本中系统评估。

这套写法能保留方向完整性，但不会越界声称未完成验证的结果。

### 5.4 若后续要把它升级为正式主贡献，需要怎样重构

Qwen 提出的风险是成立的：如果只是“多层 loss 加权蒸馏 SAM2”，很容易被审稿人视为常规工程堆叠。因此，若未来要把这条线升级成主文核心创新，必须引入更有辨识度的机制，而不是只写普通 distillation。一个更合理的方向是：

1. **Uncertainty-Weighted Geometric Distillation**
   使用 SAM2 在边界附近的置信度或稳定性估计，对蒸馏损失进行动态加权，降低教师不可靠区域的负迁移。
2. **Cross-Scale Geometric Alignment**
   不直接对齐 backbone feature，而是将学生 neck 中的 Shared-DT 几何场与教师几何结构先验对齐，使蒸馏服务于本文已有的几何主线。
3. **Self-Correcting Pseudo-Mask Refinement**
   让学生模型在训练过程中反向修正教师伪掩膜，而不是单向接受蒸馏，以避免 foundation model 误差被硬性继承。

只有当蒸馏机制与本文“几何驱动轻量分割”主线深度耦合，并体现出新机制而非新教师来源时，它才更有机会成为能过审的主创新点。

### 5.5 当前可保留的正式学术表述

在当前阶段，可以把这一点写成如下形式：

**作为对主体框架的扩展，我们进一步引入一种 SAM2-assisted geometric supervision strategy，在训练阶段利用高质量教师掩膜构造边界、距离变换和软掩膜监督，为轻量学生网络提供更精细的几何先验。该策略仅在训练时使用，不增加推理阶段的模型复杂度。**

注意，这里应避免写“we demonstrate”或“experiments verify”之类已经完成验证的措辞。

---

## 7. 统一损失函数与整体框架表述

基于以上设计，当前主体框架的总损失可以写为：

$$
L = L_{det} + L_{seg} + \lambda_{dt}L_{dt} + \lambda_{nwd}L_{assign}^{nwd},
$$

其中：

1. $L_{det}$ 与 $L_{seg}$ 分别表示 RTMDet-Ins 原有的检测与实例分割损失。
2. $L_{dt}$ 表示 Shared-DT DT-FPN 的几何监督损失。
3. $L_{assign}^{nwd}$ 并不是额外显式损失项，而是表示 NWD 对动态分配过程的替换作用，其影响最终体现在主任务损失的样本构成与权重计算上。

若在后续引入 SAM2 扩展训练策略，则总目标可写为：

$$
L = L_{det} + L_{seg} + \lambda_{dt}L_{dt} + \lambda_{sam}L_{sam}^{geo},
$$

其中 $L_{sam}^{geo}$ 表示基于教师边界、教师 DT 和教师软掩膜构造的几何蒸馏监督。

这里建议在写作中把 NWD 放在“训练分配机制”章节描述，而不是强行把它并入统一损失公式，以保持叙事准确。

---

## 8. 论文中可直接使用的方法章节初稿

下面给出一版可直接改写进论文正文的方法章节中文初稿。

### 7.1 问题定义

密集砾石实例分割面临三个核心挑战：其一，砾石目标普遍尺寸较小，且尺度分布高度不均衡，导致基于 IoU 的训练样本分配对微小位置偏移极其敏感；其二，相邻实例之间常出现接触、遮挡与弱边界，标准特征金字塔在跨层融合时容易造成实例间特征混叠；其三，自建砾石数据集中的掩膜标注不可避免地存在边界粗糙与监督噪声，这对轻量模型尤其不利。因此，本文旨在在轻量化约束下提升模型对密集邻接实例的定位稳定性与边界分离能力。

### 7.2 方法总览

本文以 RTMDet-Ins tiny 为基础框架，提出一种面向密集砾石实例分割的几何驱动轻量化方法。具体而言，我们从特征融合与训练分配两个层面进行针对性改造：一方面，设计 Shared-DT DT-FPN，在 top-down 多尺度融合过程中引入受显式距离变换监督约束的共享几何先验，以增强实例边界区域的结构保持能力；另一方面，在动态软标签分配中引入 NWD 相似度，以缓解小目标场景下 IoU 对微小位置偏移过于敏感的问题。此外，考虑到密集砾石场景同时需要局部边界保持和长程结构建模，本文还保留了一条轻量化 VMamba 主干探索分支，优先考虑更轻的 2241 设定以及仅替换后两个 stage 的 hybrid 方案。进一步地，本文还设计了一个 SAM2 辅助几何监督扩展，用于在训练阶段利用高质量教师掩膜构造更细致的边界与距离场监督，但该部分作为扩展训练策略保留，不影响主体框架的推理复杂度。

### 7.3 Shared-DT DT-FPN

标准 FPN 类 neck 通常对不同空间位置采用一致的特征融合规则，难以适应密集邻接实例场景下中心区域与边界区域对语义信息和细节信息的差异化需求。为此，本文引入 Shared-DT DT-FPN。不同于逐层独立预测引导图的局部注意力形式，我们首先从高层到低层迭代解码得到一张受距离变换真值监督的共享 DT 图，并在各个被引导的 top-down 融合层对其进行尺度变换。随后，该共享几何图被用于生成空间门控，以在实例内部区域增强高层语义，在邻接边界区域保留更多浅层结构细节。由于所有融合层共享同一张几何先验图，该设计能够在不同尺度上保持一致的几何语义解释，并避免逐层独立门控所带来的局部不一致问题。

### 7.4 NWD-based Dynamic Assignment

对于密集小目标，IoU 在训练初期常因预测框和真实框之间的微小位移而急剧下降，从而导致动态样本分配不稳定。针对这一问题，本文在 RTMDet-Ins 的 Dynamic Soft Label Assigner 中使用 NWD similarity 替换默认 IoU 计算。我们将边界框建模为二维高斯分布，并利用高斯中心与尺度差异构造 Wasserstein-based 几何相似度。该度量在框尚未重叠但已经接近目标中心位置时仍然具有连续响应，因此更适合密集小目标的早期训练阶段。需要指出的是，本文并不将 NWD 作为新的边界框回归损失，而仅将其用于样本分配环节，以便隔离并验证几何相似度对训练稳定性的影响。

### 8.5 Lightweight VMamba Backbone Exploration

除几何融合与训练分配之外，本文还进一步保留了一条轻量化结构替换分支，用于探索 RTMDet-Ins 主干是否能够通过部分引入 VMamba 获得更优的长程关系建模能力。考虑到纯状态空间主干在浅层高分辨率区域可能损伤细粒边界信息，而密集砾石实例分割又高度依赖局部接缝与边界保持，本文更倾向于采用一种 stage-wise heterogeneous design：保留前两阶段卷积特征提取，仅将后两个 stage 替换为轻量化 VMamba。该设计能够在浅层继续利用卷积对边缘与细缝的局部建模能力，同时在深层引入长程依赖建模能力。出于轻量化叙事的自洽性考虑，本文优先推荐比 2292 更轻的 2241 设定，并将该结构分支作为候选增强项进行后续实验验证，而不将其写成当前已经完成闭环证实的主体贡献。

### 8.6 SAM2-assisted Geometric Supervision Extension

考虑到密集砾石场景中人工标注边界的不稳定性，本文进一步设计了一种 SAM2 辅助几何监督扩展。该扩展在训练阶段利用高质量教师掩膜构造边界图、距离变换图和软掩膜监督，以为轻量学生模型提供更细粒度的几何先验。由于该策略仅在训练阶段引入教师信息，因此不会增加推理阶段的参数量和计算量。在当前工作中，我们将其作为主体框架的可扩展增强方向，而本文的核心验证重点仍然放在 Shared-DT DT-FPN 与 NWD-based assignment 上。

---

## 9. 各创新分支的正式学术表述建议

下面给出更适合写在摘要、引言贡献点和方法章节末尾的小结版本。

### 版本 A：当前最稳妥、适合投稿主文的三点写法

1. We propose a shared distance-transform guided feature pyramid for lightweight RTMDet-Ins, where a globally decoded DT prior is reused across top-down fusion stages to explicitly regulate the trade-off between semantic abstraction and boundary preservation in dense gravel scenes.
2. We integrate NWD-based similarity into the dynamic assignment process of RTMDet-Ins to alleviate the instability of IoU-driven sample matching for dense tiny instances, while keeping the regression objective unchanged for fair isolation of assignment effects.
3. We further develop a SAM2-assisted geometric supervision extension that transfers high-quality boundary and distance-field priors to the lightweight student during training, without increasing inference-time complexity.

若后续 VMamba 分支实验成立，则可额外加入：

4. We further explore a lightweight heterogeneous backbone design that preserves convolutional early stages while replacing late stages with a VMamba-2241 style state-space encoder, aiming to balance local boundary preservation and long-range structural modeling.

### 版本 B：更保守、风险更低的贡献点写法

1. We propose a geometry-driven lightweight RTMDet-Ins framework for dense gravel instance segmentation.
2. We design a shared-DT neck fusion mechanism and an NWD-based assignment strategy to improve boundary-aware representation and small-object training stability, respectively.
3. We discuss a SAM2-assisted geometric supervision extension as a promising training enhancement for future versions of the framework.

对于当前阶段，建议优先使用版本 B 写在摘要和引言贡献点中；版本 A 可用于内部汇报、开题、中期答辩或实验补全后的升级稿。

---

## 10. 最终推荐的论文主线

综合当前进度与风险，本文最推荐的主线如下：

1. **主文方法：Shared-DT DT-FPN + NWD assignment。**
2. **结构候选分支：Hybrid VMamba-2241 优先，其次再看 Full VMamba-2241；2292 只作为容量参考，不作为轻量化主卖点。**
3. **主文实验：轻量模型对比 + 自身消融 + 效率分析。**
4. **SAM2：补充实验、讨论节或后续版本增强项。**

一句话概括整篇论文：

**本文不是通过堆叠更重的模块去提升砾石实例分割，而是通过显式几何先验驱动的 neck 融合、小目标友好的动态分配，以及可选的轻量化 VMamba 结构替换，使 RTMDet-Ins 更适合密集、粘连、边界模糊的砾石场景。**

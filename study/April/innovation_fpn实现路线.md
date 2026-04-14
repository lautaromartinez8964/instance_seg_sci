# RS-LightMamba Innovation-FPN 实现路线

## 0. 文档定位

这份文档用于承接 backbone 创新完成后的下一阶段主线，也就是 FPN 侧创新的定义、实现、训练顺序和代码落地说明。

从当前阶段开始，论文主线拆分为三段：

1. Innovation 1：Stage 4 Global Attention，作为 backbone 主创新的阶段性冻结版本。
2. Innovation FPN：在 neck 侧验证更有效的多尺度融合引导机制，当前候选为 IG-FPN 和 HF-FPN。
3. Innovation 3：后续再进入 integral distillation，不与当前 neck 创新混做。

这份手册的目的不是单纯记录想法，而是固定下面四件事：

1. IG-FPN 和 HF-FPN 分别在解决什么问题。
2. 它们从 backbone 取出什么信息，如何作用到 FPN。
3. 监督如何构造，作用在哪些 stage。
4. 当前代码已经实现到什么程度，以及实验顺序如何安排。

---

## 1. 当前阶段结论

### 1.1 Backbone 主线先冻结

当前 backbone 主线已经不再继续围绕旧版 reorder 或 dt z 双路调制扩展，而是先以 Stage 4 Global Attention 为主干冻结。

这一阶段的关键判断已经明确：

1. 只替换最终 Stage 4 为 global attention 后，实验已经显示出有效增益。
2. 这说明当前更值得继续深挖的是 neck 端如何把浅层细节和高层语义融合得更好，而不是继续把 backbone 改得越来越重。
3. 因此，innovation1 可以视为完成阶段性收束，下一步进入 innovation_fpn。

### 1.2 Innovation-FPN 的核心问题

当前的标准 FPN 仍然是“无差别地”做 top-down 融合：

$$
P_{i-1} = L_{i-1} + Up(P_i)
$$

它默认所有空间位置都应该被同等融合，但遥感实例分割并不满足这个假设：

1. 前景稀疏，小目标只占很少区域。
2. 背景中存在大量纹理和伪边界。
3. 小目标真正依赖的是“哪些区域值得传递语义”和“哪些边缘值得保留细节”。

所以当前 neck 创新的本质，是把原来的“均匀融合”改成“有引导的融合”。

---

## 2. 为什么在 IG-FPN 和 HF-FPN 之间做抉择

这两个候选方案本质上代表两种不同的指导信号来源：

1. IG-FPN：让 FPN 受到目标相关语义重要性引导。
2. HF-FPN：让 FPN 受到浅层高频结构先验引导。

它们都遵循同一个大框架：

$$
P_{i-1} = L_{i-1} + Up(P_i) \cdot G_{i-1}
$$

区别只在于门控项 $G_{i-1}$ 来自哪里。

### 2.1 IG-FPN 的门控来源

IG-FPN 的门控来自一张共享 importance map：

$$
G_{i-1}^{IG} = 1 + \alpha_{i-1} \cdot Imp_{i-1}
$$

这里的 $Imp_{i-1}$ 来自 S3 和 S4 的融合预测，因此它是“语义驱动”的引导。

### 2.2 HF-FPN 的门控来源

HF-FPN 的门控来自每个浅层 stage 的单通道高频图：

$$
G_{i-1}^{HF} = 1 + \beta_{i-1} \cdot HF_{i-1}
$$

这里的 $HF_{i-1}$ 来自固定拉普拉斯高通滤波，因此它是“结构驱动”的引导。

### 2.3 为什么先训 IG-FPN

当前优先级定为先训 IG-FPN，再训 HF-FPN，原因如下：

1. IG-FPN 与此前的 IG 系列研究线更连贯，叙事上延续性最好。
2. IG-FPN 的 guidance map 有显式前景监督，可解释性更强。
3. 当前目标是先拿一个稳定增益版本，而不是同时追两个复杂变量。
4. 如果 IG-FPN 已经带来提升，就能证明“语义重要性引导 FPN”本身成立；随后再用 HF-FPN 回答“纯高频结构先验能不能更强”。

---

## 3. 统一的 Backbone 到 Neck 接口定义

### 3.1 当前固定的 backbone 输出

在当前主线中，backbone 正常输出四层特征：

1. Stage 1 对应 C2
2. Stage 2 对应 C3
3. Stage 3 对应 C4
4. Stage 4 对应 C5

在代码里，这四个 stage 的索引是 0, 1, 2, 3，对应关系如下：

| 论文记号 | 代码 stage 索引 | FPN 输入名 | 通道数 |
|---|---:|---|---:|
| Stage 1 | 0 | C2 | 96 |
| Stage 2 | 1 | C3 | 192 |
| Stage 3 | 2 | C4 | 384 |
| Stage 4 | 3 | C5 | 768 |

### 3.2 为什么 detector 不需要额外改 extract_feat

当前实现没有去改标准 TwoStageDetector 的特征提取流程，而是把额外引导信息直接打包在 backbone 输出里：

1. IG-FPN 路线下，backbone 返回二元组：features, guidance_map。
2. HF-FPN 路线下，backbone 返回二元组：features, hf_maps。
3. neck 内部自行解析这个二元组，并把特征和引导图拆开。

这意味着训练和推理主链保持干净：

1. 标准多尺度特征接口不变。
2. neck 负责解释额外 guidance。
3. detector 只额外负责把前景监督送进 backbone。

---

## 4. IG-FPN 方法定义

## 4.1 目标

IG-FPN 解决的问题是：

标准 FPN 在 top-down 传递高层语义时，不知道哪些区域是真正值得强调的目标区域，因此容易把背景语义也一并扩散到浅层。

IG-FPN 的目标是：

让 top-down 融合在空间上变得有选择性，优先增强“更可能是前景目标”的区域。

### 4.2 为什么不用四个 stage 一起预测 importance

当前明确不采用 S1 到 S4 全部采样做 importance 融合，原因如下：

1. S1 和 S2 太浅，语义弱，噪声高。
2. 如果把浅层纹理直接混入 importance 预测，很容易把 gate 污染成“高响应纹理图”，而不是“目标相关图”。
3. 当前目标是稳增益，不是堆复杂度。
4. S3 提供足够的实例细节，S4 提供足够的全局语义，这一对组合已经足够构成高质量 guidance。

因此 IG-FPN 的第一版固定为：

1. 用 S3 和 S4 预测同一张共享 guidance map。
2. 先只作用在 P3 和 P2。
3. 后续若效果明确，再考虑是否扩展作用层数，而不是先扩展预测输入层数。

### 4.3 Backbone 侧输出什么

IG-FPN 路线下，backbone 正常输出四层特征 C1, C2, C3, C4，同时额外输出一张由 S3 和 S4 融合得到的单通道 importance map。

具体定义为：

$$
	ilde{S}_4 = Up(Proj_4(S_4) \rightarrow S_3\ size)
$$

$$
Imp = \sigma \left( Head([Proj_3(S_3), \tilde{S}_4]) \right)
$$

其中：

1. $Proj_3$ 和 $Proj_4$ 是 1x1 卷积投影。
2. 先把 S4 上采样到 S3 尺度。
3. 然后与 S3 的投影特征拼接。
4. 最后送入一个轻量 ForegroundHead 得到单通道 importance map。

当前代码实现对应：

1. 在 backbone 中新增 CrossStageGuidanceHead。
2. guidance_stages 固定为 [2, 3]，即论文里的 Stage 3 + Stage 4。
3. 输出分辨率与 S3 一致，也就是输入 256 时对应 16x16。

### 4.4 IG-FPN 如何监督

IG-FPN 的 guidance map 采用显式前景监督。

监督构造方式如下：

1. 从每张图的所有实例 mask 合成一张二值前景图。
2. 如果前景图尺寸与 guidance map 不一致，就用 area 插值下采样到 guidance 尺度。
3. 使用 BCE 作为辅助监督。

数学上写成：

$$
L_{fg}^{IG} = \lambda_{ig} \cdot BCE(Imp, FG)
$$

这里的监督路径是：

1. detector 在训练时把 gt masks 合并成前景图。
2. backbone 内部缓存前景目标。
3. guidance head 输出后直接在 backbone 内部计算 loss_fg。
4. 最终由 detector 一并把 loss_fg 合到总损失里。

因此 IG-FPN 不是无监督 gate，而是显式前景引导的 gate。

### 4.5 IG-FPN 如何作用到 FPN

IG-FPN 在 top-down 融合时复用同一张 guidance map，并按目标层分辨率进行 resize：

$$
Imp_{i-1} = Resize(Imp, H_{i-1}, W_{i-1})
$$

$$
P_{i-1} = L_{i-1} + Up(P_i) \cdot (1 + \alpha_{i-1} \cdot Imp_{i-1})
$$

这里每一层都有一个可学习标量 $\alpha_{i-1}$，并且初值为 0。

这一步非常关键，因为它保证：

1. 训练初始时，IG-FPN 自动退化为标准 FPN。
2. 模型不会在一开始就被错误 gate 破坏。
3. 如果 importance map 学得有效，alpha 会逐步偏离 0，开始提供正向引导。

### 4.6 当前 IG-FPN 只作用在哪些层

当前第一版 IG-FPN 只作用在 P3 和 P2，也就是对小目标最敏感的两层。

代码里对应 guided_levels = [0, 1]，含义是：

1. 对 C3 融合出的 P3 使用 gate。
2. 对 C2 融合出的 P2 使用 gate。
3. 对更高层 P4 暂时不加 gate。

这样设计的原因是：

1. 小目标和细边界最依赖 P2 P3。
2. 在第一版里先限制 gate 的作用范围，更容易拿到稳定结论。
3. 如果一开始全层 gate，出现退化时不容易判断是 guidance 质量问题还是作用层数过多。

### 4.7 IG-FPN 的论文表述

IG-FPN 可以定义为：

一种由跨阶段语义重要性图引导的 selective top-down fusion 机制。它利用 Stage 3 的局部细节与 Stage 4 的高层语义共同生成共享 importance map，并在 FPN 的浅层 top-down 融合阶段对上采样特征进行位置相关的自适应放缩，从而增强目标相关区域的语义注入强度，同时抑制背景语义污染。

---

## 5. HF-FPN 方法定义

### 5.1 目标

HF-FPN 解决的问题与 IG-FPN 不同。

它并不强调“哪里更像前景”，而是强调“哪里具有更强的高频结构或边界信息”。

它的出发点是：

1. 遥感实例分割尤其是小目标，往往依赖细边界和局部纹理差异。
2. 标准 FPN 在 top-down 融合时可能把高层语义带下来，但不能显式强调这些高频位置。
3. 因此可以用浅层高频图来门控 top-down 融合，让高频区域获得更强的语义注入。

### 5.2 Backbone 侧输出什么

HF-FPN 路线下，backbone 正常输出四层特征 C2, C3, C4, C5，同时额外输出三张单通道高频图，分别对应 Stage 1, Stage 2, Stage 3。

高频图来自固定拉普拉斯高通滤波：

$$
F_k = Laplace(C_k)
$$

$$
HF_k = \frac{Mean_c(|F_k|)}{\max(Mean_c(|F_k|)) + \epsilon}
$$

这里的实现要点是：

1. 先对每个通道做固定拉普拉斯卷积。
2. 对通道维取绝对值平均，得到单通道能量图。
3. 再按样本做最大值归一化，避免不同 batch 的幅值不可比。

这意味着 HF-FPN 不依赖额外可学习的高频分支，而是先用固定算子构造一个简洁的高频先验。

### 5.3 HF-FPN 如何作用到 FPN

HF-FPN 在每一层 top-down 融合时使用对应尺度的高频图门控上采样特征：

$$
P_{i-1} = L_{i-1} + Up(P_i) \cdot (1 + \beta_{i-1} \cdot HF_{i-1})
$$

其中每层有一个可学习标量 $\beta_{i-1}$，初值同样为 0。

当前实现对应关系是：

1. Stage 1 的高频图用于 P2 融合。
2. Stage 2 的高频图用于 P3 融合。
3. Stage 3 的高频图用于 P4 融合。

所以 HF-FPN 的第一版默认是全 top-down 三层都可被 gate。

### 5.4 HF-FPN 如何监督

HF-FPN 当前没有显式额外监督。

也就是说：

1. 高频图本身由固定滤波器直接生成。
2. 只有门控强度参数 $\beta$ 是可学习的。
3. 是否真的有用，由主任务损失反向决定。

这使 HF-FPN 的优点和风险都很清晰：

1. 优点是实现非常轻，不额外引入复杂监督路径。
2. 风险是高频不一定等于前景，背景纹理也可能被强化。

### 5.5 HF-FPN 的论文表述

HF-FPN 可以定义为：

一种基于浅层高频响应的 gated top-down fusion 机制。它从浅层 backbone 特征中提取归一化的单通道高频图，并将其作为位置相关的结构先验，调制来自高层语义特征的上采样分支，从而在多尺度融合过程中显式保留目标边缘和局部几何细节。

---

## 6. IG-FPN 与 HF-FPN 的核心差异

| 维度 | IG-FPN | HF-FPN |
|---|---|---|
| 指导信号 | S3+S4 融合 importance map | S1+S2+S3 高频图 |
| 信息属性 | 语义前景引导 | 结构边缘引导 |
| 是否有显式监督 | 有，BCE 前景监督 | 无，仅靠主任务反传 |
| 预测复杂度 | 有一个轻量 guidance head | 无额外预测头，固定滤波 |
| 第一版 gate 作用层 | P3, P2 | P4, P3, P2 |
| 主要收益预期 | 减少背景语义污染 | 强化边缘与细节保留 |
| 主要风险 | guidance 若学偏会产生错误门控 | 高频噪声会被误增强 |

一句话概括：

1. IG-FPN 更偏“目标相关区域增强”。
2. HF-FPN 更偏“边界细节区域增强”。

---

## 7. 当前代码实现映射

### 7.1 Backbone 侧实现

当前 backbone 实现统一放在 mmdet/models/backbones/rs_lightmamba/lightmamba_backbone.py。

其中与 FPN 创新直接相关的模块有两类：

1. CrossStageGuidanceHead
	- 用于 IG-FPN
	- 输入 S3 和上采样后的 S4
	- 输出单通道 guidance map

2. SpatialHighFrequencyExtractor
	- 用于 HF-FPN
	- 对 Stage 1 到 Stage 3 特征做固定拉普拉斯高通
	- 输出单通道高频图

backbone 当前新增了两组可选输出开关：

1. output_guidance_map
2. output_hf_maps

对应的 forward 返回形式为：

1. 标准 backbone：只返回 tuple(features)
2. IG-FPN backbone：返回 tuple(features), guidance_map
3. HF-FPN backbone：返回 tuple(features), tuple(hf_maps)

### 7.2 监督是如何接进去的

当前 detector 实现位于 mmdet/models/detectors/rs_ig_mask_rcnn.py。

训练时的路径是：

1. loss 开始前，detector 调用 backbone.set_ig_targets。
2. backbone 将 gt instance masks 合成前景图。
3. guidance map 或 attention importance 产生后，在 backbone 内部计算辅助 BCE。
4. detector 再调用 backbone.get_ig_aux_losses，把 loss_fg 合并到总损失里。

这条路径意味着：

1. IG-FPN 的 guidance supervision 已经是完整闭环。
2. HF-FPN 没有额外监督，因此只会返回普通主任务损失。

### 7.3 Neck 侧实现

当前 neck 实现分成两个文件：

1. mmdet/models/necks/ig_fpn.py
2. mmdet/models/necks/hf_fpn.py

两者都继承自 MMDetection 原生 FPN，只改 forward 中的 top-down 融合部分。

共同点：

1. 都保留原始 lateral conv 和 fpn conv。
2. 都在融合时构造 gate = 1 + scale x guidance。
3. 都把缩放参数初始化为 0，从标准 FPN 起步。

不同点：

1. IG_FPN 的输入是 features 加一张 guidance_map。
2. HF_FPN 的输入是 features 加一个 hf_maps 列表。

### 7.4 配置文件对应关系

当前相关配置已经明确：

1. Stage 4 Global Attention 基线
	- projects/iSAID/configs/mask_rcnn_rs_lightmamba_s4_global_attn_fpn_1x_isaid.py

2. IG-FPN
	- projects/iSAID/configs/mask_rcnn_rs_lightmamba_s4_global_attn_ig_fpn_1x_isaid.py

3. HF-FPN
	- projects/iSAID/configs/mask_rcnn_rs_lightmamba_s4_global_attn_hf_fpn_1x_isaid.py

这三者的关系应该固定理解为：

1. 第一份是 neck 创新的共同 backbone 基线。
2. 第二份是在该 backbone 基线上切入 IG-FPN。
3. 第三份是在同一 backbone 基线上切入 HF-FPN。

因此后续对比时，必须保证 backbone 主线一致，只比较 neck 创新本身。

---

## 8. 当前实验顺序固定

为了保证结论可解释，当前 neck 创新实验顺序固定如下：

### 8.1 第一步：先训 IG-FPN

IG-FPN 是当前第一优先级，原因已经固定：

1. 它最接近此前的 IG 系列研究叙事。
2. 它的 guidance map 有明确监督，结论更好解释。
3. 它当前只 gate P3 P2，风险最低。

因此本阶段的首个实验应定义为：

在 Stage 4 Global Attention backbone 基线上，接入 S3+S4 融合 guidance map，并在 P3 P2 的 top-down 融合中进行门控。

### 8.2 第二步：再训 HF-FPN

HF-FPN 是第二个对比实验，用来回答：

如果不用显式语义 guidance，而只用高频结构先验，是否也能带来稳定收益。

这一步的意义是：

1. 给 IG-FPN 一个强对照。
2. 判断语义引导和结构引导，哪个更适合当前 iSAID 主线。
3. 为后续论文中 neck 创新部分提供二选一主表依据。

### 8.3 现阶段不做 IG 加 HF 混合版

当前不建议直接做 IG-FPN 与 HF-FPN 的混合版，原因如下：

1. 现在最重要的是先判断哪条路本身有效。
2. 如果一开始就混合，两者的贡献无法拆解。
3. 混合版应当只在其中一条主线明确有效后，再作为后续追加实验。

---

## 9. 当前已完成的代码状态

截至当前版本，innovation_fpn 相关工程状态如下：

1. IG-FPN 已完成代码实现。
2. HF-FPN 已完成代码实现。
3. 两条分支都已经完成构建和 forward 级联验证。
4. IG-FPN 已被设为当前优先训练对象。

其中 IG-FPN 的当前具体设定为：

1. guidance_stages = [2, 3]
2. guidance_hidden_dim = 256
3. guidance_use_fg_loss = True
4. guidance_loss_weight = 0.2
5. guided_levels = [0, 1]

而 HF-FPN 的当前具体设定为：

1. hf_map_stages = [0, 1, 2]
2. 每层 gate 强度参数 beta 初始化为 0
3. 默认覆盖 P4, P3, P2 三个 top-down 融合位置

---

## 10. 论文写作时的推荐叙事

在论文中，建议不要把 IG-FPN 和 HF-FPN 同时写成两个并列主创新，而是写成一个 innovation_fpn 模块下的两种候选设计，并最终选择其中效果更强的一条作为正式主模块。

推荐写法：

1. 先提出问题：标准 FPN 缺少空间选择性融合。
2. 再提出两种可行的指导信号来源：语义重要性与高频结构。
3. 通过实验比较两者。
4. 选择效果更好的那个写入主方法。
5. 另一个作为补充对照或附录版本。

这样的好处是：

1. 论文结构更干净。
2. 实验逻辑更清楚。
3. 不会让审稿人觉得 neck 部分在堆模块。

---

## 11. 当前阶段结论与下一动作

当前结论已经固定：

1. backbone 主线以 Stage 4 Global Attention 为阶段性冻结版本。
2. innovation_fpn 的比较对象固定为 IG-FPN 与 HF-FPN。
3. IG-FPN 当前采用 S3+S4 融合 guidance，而不是四个 stage 全采样。
4. 第一轮训练先跑 IG-FPN，再用同 backbone 基线跑 HF-FPN 做对照。

下一阶段只做两件事：

1. 记录 IG-FPN 的完整训练曲线与验证指标。
2. 在相同 backbone 基线下启动 HF-FPN，对比谁更适合作为 neck 主创新。

如果 IG-FPN 胜出，论文叙事就是“语义重要性引导的 selective FPN fusion”。

如果 HF-FPN 胜出，论文叙事就是“高频结构先验驱动的 gated top-down fusion”。

在这两个结果出来之前，不再引入第三种 neck 变体，避免实验面过宽。

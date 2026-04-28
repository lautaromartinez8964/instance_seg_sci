# 4_20 DT-FPN v1
我先把这条 DT-only 实现涉及的几个文件重新对齐一下，用代码级别而不是概念级别给你拆开说明：数据怎么进、DT 怎么预测、loss 怎么算、FPN 怎么被 gate。

我已经把这条 DT-only 代码链路重新串起来了，下面给你的解释会按“数据进入模型 → neck 里预测 DT → DT loss → FPN 融合怎么被改写”这条顺序展开，并把当前实现里和你手册理想版之间的差别也点出来。

可以，我按代码实际执行路径给你完整拆开讲。这一版实现的定位要先说清楚：

它是一个“DT-only 的第一步消融版”。
也就是：
不引入 IBD
不引入 boundary_map
不做多尺度 DT
只验证一件事：
如果给 FPN 一个可学习的 DT 几何先验，top-down 融合会不会比标准 FPN 更合理。

核心代码主要在这几个文件里：

dt_fpn.py  
gravel_loading.py  
formatting.py  
rs_ig_mask_rcnn.py  
mask_rcnn_gravel_lightmamba_dt_fpn_36e_gravel_big.py

**1. 数据是怎么进模型的**
你现在已经有 DT 真值，训练配置里直接把它接进了 pipeline。入口在 mask_rcnn_gravel_lightmamba_dt_fpn_36e_gravel_big.py。

训练 pipeline 里多了一步 `LoadGravelDistanceTransform`，也就是在读完图像和实例标注之后，再去读离线生成好的 DT 图。具体读取逻辑在 gravel_loading.py。

这个 transform 干了三件事：

1. 根据当前图像路径自动推导 split 和文件名。
2. 到 `distance_root/split/distance_transform/图像名.png` 去找对应的 DT 图。
3. 把它塞进 `results['gt_seg_map']`。

这里有一个非常关键的工程取舍：  
我没有新造一个 `gt_dt_map` 字段走全新增强链，而是故意复用了 `gt_seg_map` 这条现成语义图通路。

为什么这么做？
因为 MMDetection 现有的 `Resize`、`RandomFlip`、`RandomChoiceResize` 这些几何增强，本来就会自动同步处理 `gt_seg_map`。这样你的 DT 图就会和图像保持严格的几何对齐，不需要我再去魔改一堆基础增强类。

接着在 formatting.py 里，`PackDetInputs` 会把这个 `gt_seg_map` 打进 `data_sample.gt_sem_seg.sem_seg`。
也就是说，到了模型内部，DT 真值就不再叫 “dt map”，而是暂时借道成了一张语义图张量。

这一层的张量形态大致是：

如果原始 DT 图是单通道二维图：
`H x W`

打包后会变成：
`1 x H x W`

然后跟着 batch 堆叠，最终 neck 读到的是：
`B x 1 x H x W`

**2. detector 是怎么把 DT target 交给 neck 的**
这个逻辑在 rs_ig_mask_rcnn.py。

我在这个 detector 里新加了一套 neck auxiliary target 生命周期，跟之前 backbone auxiliary loss 的思路一样：

1. `loss()` 开始前先调用 `_set_neck_auxiliary_targets`
2. neck 自己缓存当前 batch 的 DT 真值
3. 正常跑 two-stage detector 的 forward 和检测损失
4. 最后再通过 `_get_neck_auxiliary_losses` 把 `loss_neck_dt` 取出来并并入总 loss

所以训练总损失现在不是只有原始 Mask R-CNN 的：

RPN 分类回归  
ROI 分类回归  
Mask loss

还会额外加一项：

`loss_neck_dt`

这项损失完全由 neck 自己维护，detector 只是负责在合适的时候把 target 塞进去、把 loss 取出来。

**3. DTFPN 里怎么预测 DT**
核心实现在 dt_fpn.py。

这个类是直接继承标准 FPN 的，所以整体骨架没有变，还是那套：

1. backbone 输出多层特征
2. 每层过 lateral 1x1 conv
3. 深层向浅层 top-down 融合
4. 每层再过 3x3 fpn conv 输出金字塔特征

真正改的是两部分：
一部分是 DT 预测头
一部分是 DT 引导的融合规则

先说 DT 预测头。

在代码里，所有 backbone 特征先过 lateral conv，得到 `laterals`。
然后直接取最深那一层：

`laterals[-1]`

也就是最深层的 lateral feature。

然后过一个很轻量的 head：

3x3 conv  
ReLU  
1x1 conv

具体就是：

输入通道：`out_channels`，这里是 256  
中间通道：`dt_head_channels`，当前配置是 128  
输出通道：1

最后再过一个 `sigmoid`：

$$
dt\_map = \sigma(\text{dt\_head}(laterals[-1]))
$$

所以这张 `dt_map` 的数值范围是 `[0,1]`。

它的含义不是硬编码写死的“1 一定是中心、0 一定是边界”，而是通过和真值做 MSE 学出来的。因为你的 GT distance transform 本身就是归一化到 `[0,1]` 的，所以网络会被逼着学成：

实例中心附近值更高  
接近边界的地方值更低  
背景或边缘弱区域值更低

但这里要注意一点：  
当前实现里 DT 只从最深层 lateral 预测一次，是单尺度、粗粒度的 DT 估计，不是多尺度 DT。

**4. DT 真值在 neck 里是怎么拿到的**
还是在 dt_fpn.py。

neck 在 `set_auxiliary_targets()` 里遍历 batch 的 `data_sample`，从每个 sample 里读出：

`data_sample.gt_sem_seg.sem_seg`

然后 `_extract_dt_target()` 做了一些防御性处理：

1. 保证维度至少是 `1 x H x W`
2. 如果值域还是 `0~255`，就除以 255 归一化到 `0~1`
3. 最后 clamp 到 `[0,1]`

然后把整个 batch 堆叠成：

`B x 1 x H x W`

保存到：

`self._current_dt_target`

这个成员变量里。

所以 neck 预测 DT 的时候，GT 已经提前缓存好了，不需要 forward 时再去碰 data sample。

**5. DT loss 是怎么写的**
这部分在 dt_fpn.py 里的 `_compute_dt_loss()`。

逻辑很直接：

1. 拿当前预测的 `dt_map`
2. 拿缓存好的 GT：`self._current_dt_target`
3. 如果 GT 分辨率和预测分辨率不一致，就把 GT 下采样到预测尺寸
4. 做 MSE
5. 再乘一个权重 `dt_loss_weight`

数学形式就是：

$$
L_{dt} = \lambda_{dt} \cdot \text{MSE}(dt\_pred, dt\_gt^{\downarrow})
$$

你现在配置里的 `dt_loss_weight=0.2`，在 mask_rcnn_gravel_lightmamba_dt_fpn_36e_gravel_big.py 里。

这里的一个关键细节是：
GT 是向预测尺寸对齐，不是反过来把预测上采样到 GT 尺寸。
也就是说，当前 loss 监督的是“粗尺度 DT 是否合理”，不是要求最深层直接恢复原图级精细 DT。
这和当前单尺度实现是一致的。

**6. FPN 阶段到底是怎么起作用的**
这是最核心的部分。

标准 FPN 的 top-down 融合其实非常简单，本质就是：

$$
P_l = L_l + U(P_{l+1})
$$

也就是：
当前层 lateral 特征 `L_l`
加上上一级上采样过来的深层特征 `U(P_{l+1})`

问题在于，这个融合是全空间统一的。
边界区域、实例中心、稠密小石子、大片石块，统统一样加。

而 DT-FPN 改的就是这里。

在 dt_fpn.py 里，top-down 还是从深层往浅层循环，但每一层融合前，会先把这张最深层预测得到的 `dt_map` 上采样到当前层大小：

$$
\hat d_l = \text{Interp}(dt\_map, size=P_l)
$$

然后再用每层自己的可学习参数 `gate_scales[l]` 和 `gate_biases[l]` 生成一个 gate：

$$
g_l = \sigma(\alpha_l \cdot \hat d_l + \beta_l)
$$

这里：
$\alpha_l$ 对应 `gate_scales`
$\beta_l$ 对应 `gate_biases`

然后不是简单相加，而是做加权插值：

$$
P_l = (1-g_l)\cdot L_l + g_l \cdot U(P_{l+1})
$$

这个式子就是当前实现的核心。

它的直觉是：

1. 如果某个位置 DT 值高
说明更像实例中心区域
那么 gate 更大
这时更偏向深层语义特征

2. 如果某个位置 DT 值低
说明更靠近边界
那么 gate 更小
这时更偏向本层浅层 lateral 特征

所以它不是“再加一张图做监督”，而是真的把 FPN 融合规则改掉了。

**7. 这三个 guided_levels 是怎么回事**
你配置里现在写的是：

`guided_levels=[0, 1, 2]`

在 mask_rcnn_gravel_lightmamba_dt_fpn_36e_gravel_big.py。

这个意思是：
三次 top-down 融合全部受 DT gate 控制。

如果 backbone 给四层特征：
C2, C3, C4, C5

那么 FPN 的 top-down 是：

C5 -> C4  
C4 -> C3  
C3 -> C2

这三个 target level 都被 DT 引导。

如果某一层不在 `guided_levels` 里，就退化回标准 FPN 的普通相加。

所以这个参数本质上决定了：
DT 到底要控制多浅的层。

**8. 当前实现和你手册里的理想版差在哪**
这部分必须说清楚，不然你容易把“现在代码实现了什么”和“我们想做什么”混成一件事。

当前代码实现的是：

1. 单尺度 DT 预测
只从最深层 lateral 预测一张 DT 图

2. 纯 DT 引导
没有显式使用 IBD 的 boundary_map

3. 纯 top-down gating
没有额外做 scale-aware 半径推断，也没有多尺度 DT head

4. 简单 MSE 监督
没有更复杂的多尺度 loss 或 teacher distillation

所以它只是你手册里 DT-FPN 的最小起步版，不是最终完整版。

你手册里的更完整版本其实应该逐步走向：

1. 多尺度 DT
而不是只从最深层出一张图

2. boundary-aware gate
也就是把 IBD 的边界图并进 gate，而不是只靠 DT

3. scale-aware DT-FPN
不同层对 DT 的解释不同，或者直接让 DT 携带实例尺度偏置

但这些都应该是后续 ablation，不该一口气堆到第一版里。

**9. 为什么我会把 DT 真值塞进 gt_seg_map，而不是单独建 gt_dt_map**
这个你应该也关心。

原因非常务实：
当前这条 DT-only 实验线不带 IBD，所以 `gt_seg_map` 没有被 boundary supervision 占用。
这时拿它来承载 DT 图最省事，而且最安全。

好处有三个：

1. 所有现成几何增强自动生效
resize、flip 都不用你额外改

2. PackDetInputs 自动打包
不用再去改 data sample 结构

3. neck 直接从 gt_sem_seg 读取
detector 侧只补一套 auxiliary target 生命周期就够了

这其实是一个工程简化手段，不是理论绑定。
等你以后要把 DT 和 IBD 同时开起来，再把 DT 拆成独立字段会更合理。

**10. 你现在这版代码的完整执行顺序**
如果把整个训练前向按时间顺序写出来，大概就是：

1. 读图像和实例标注
2. 读对应的 distance transform png
3. 把 DT 图塞进 `gt_seg_map`
4. resize / flip 时，图像和 DT 图一起变换
5. PackDetInputs 把 DT 图打成 `data_sample.gt_sem_seg`
6. detector 在 `loss()` 前调用 neck 的 `set_auxiliary_targets()`
7. neck 把 batch 里的 DT GT 缓存起来
8. backbone 正常输出多层特征
9. DTFPN 先做 lateral conv
10. 用最深层 lateral 预测一张 `dt_map`
11. 用 GT 下采样后和 `dt_map` 做 MSE，得到 `loss_neck_dt`
12. 同时把 `dt_map` 上采样到各层，生成 gate
13. 用 gate 改写每一层 top-down 融合
14. 输出新的 FPN 多层特征给 RPN 和 ROI head
15. detector 把标准检测损失和 `loss_neck_dt` 一起返回

所以这条链路里，DT 同时承担两个角色：

1. 监督目标
通过 MSE 逼 neck 学出一张几何图

2. 融合先验
通过 gate 改写 FPN 的特征流

这就是它和“纯辅助监督分支”最本质的区别。

**11. 一句话总结当前这版 DTFPN**
这版代码的本质是：

从最深层 FPN 语义特征里预测一张粗粒度的 distance transform 图，用 MSE 对齐离线生成的 DT 真值，再把这张图上采样成空间 gate，控制每一级 top-down 融合在“浅层细节”和“深层语义”之间的取舍。

如果你愿意，我下一步可以继续做两件很实用的事，二选一：

1. 我把这段实现再整理成“论文方法小节”写法，直接对应你的创新手册叙事。
2. 我把当前这版 DTFPN 的 forward 用伪代码重新画一遍，让你一眼看清楚张量怎么流动。

# 4_20 DT-FPN v2
这一版是对 v1 的直接结构升级，但仍然保持“shared DT map + shared gate”这条研究主线不变。

它不再只从最深层 lateral 特征预测一张粗 DT 图，而是改成：

1. 先对每一级对齐后的 lateral 特征做通道压缩，得到统一维度的 DT 分支特征。
2. 从最深层开始，逐级上采样 DT 分支特征。
3. 每到一层，就和当前层的 lateral-DT 特征做 concat 融合。
4. 一直解码到最高分辨率层，最后输出一张共享高分辨率 DT guidance map。
5. 再像 v1 一样，把这张共享 DT map resize 到各个 FPN 融合层，用同一套 gate 逻辑去控制 top-down 融合。

也就是说，v2 改的是“DT map 怎么来”，不改“DT map 怎么用”。
这正适合作为 v1 的下一步干净消融。

## v2 的核心代码结构

在 dt_fpn.py 里，原来的：

- 单个 `dt_head`

被替换成三部分：

1. `dt_lateral_adapters`
作用：把每一级 lateral 特征都压到同一个 `dt_head_channels` 维度，作为 DT 解码器的输入。

2. `dt_fuse_blocks`
作用：在逐级上采样过程中，把“上一层解码出来的 DT 特征”和“当前层 lateral 的 DT 特征”融合起来。

3. `dt_predictor`
作用：在最高分辨率 DT 特征上输出最终的单通道共享 DT map。

## v2 的 DT 预测流程

设 FPN lateral 特征为：

- `L2, L3, L4, L5`

其中 `L5` 最深，`L2` 分辨率最高。

v2 的 shared DT 解码流程是：

1. 先从 `L5` 得到初始 DT 特征：

$$
D_5 = A_5(L_5)
$$

其中 `A_i` 表示各层的 `dt_lateral_adapter`。

2. 然后逐级向上解码：

$$
D_4 = F_4([U(D_5), A_4(L_4)])
$$

$$
D_3 = F_3([U(D_4), A_3(L_3)])
$$

$$
D_2 = F_2([U(D_3), A_2(L_2)])
$$

其中：

- `U` 表示双线性上采样
- `F_i` 表示 `dt_fuse_blocks`
- `[]` 表示通道拼接 concat

3. 最后输出共享 DT 图：

$$
dt_{shared} = \sigma(\text{Predictor}(D_2))
$$

所以 v2 输出的 `dt_shared` 已经是在最高分辨率 lateral 层上解码出来的一张共享几何图，而不再是 v1 那种只来自最深层的粗图。

## v2 的损失怎么变

这一版还没有引入多尺度 loss。

也就是说，当前 v2 仍然只对最终输出的这张共享 DT map 做监督：

$$
L_{dt}^{v2} = \lambda_{dt} \cdot \text{MSE}(dt_{shared}, dt_{gt}^{\downarrow/\uparrow})
$$

实现上和 v1 一样：

1. 读取 batch 里的 DT GT
2. 如果 GT 尺寸和 `dt_shared` 不一致，就插值到相同尺寸
3. 做 MSE
4. 乘 `dt_loss_weight`

所以 v2 仍然是“单张共享 DT 的监督”，只是这张 DT 图不再是单 deepest 预测，而是多尺度联合解码得到的。

## v2 在 FPN 阶段怎么作用

这一部分和 v1 保持一致，没有改 gate 的结构，只改了 gate 的输入质量。

对每一个被引导的 FPN level：

1. 先把 `dt_shared` resize 到当前层大小
2. 通过 `gate_scales[level]` 和 `gate_biases[level]` 生成 gate：

$$
g_l = \sigma(\alpha_l \cdot \hat d_l + \beta_l)
$$

3. 再执行软路由融合：

$$
P_l = (1-g_l) \cdot L_l + g_l \cdot U(P_{l+1})
$$

所以 v2 的实质是：

- v1：单 deepest DT prediction -> shared DT map -> shared gate
- v2：multi-scale shared DT decoder -> shared DT map -> shared gate

## v2 相比 v1 的主要改进点

1. DT 图不再只依赖最深层语义特征，而是整合了多层 lateral 的局部几何信息。
2. 最终共享 DT map 的分辨率更高，更适合 320x320 这种小图里的密集小砾石场景。
3. 这一步只替换了 DT prediction 路径，没有同时把 gate 机制改复杂，因此非常适合作为 v1 的后续消融版本。

## v2 仍然没有做的事

1. 还没有做 per-level DT map。
2. 还没有做多尺度 DT loss。
3. 还没有显式引入 IBD boundary map 去调 gate。
4. 还没有做 scale-aware DT-FPN。

所以 v2 的定位很明确：

它是“shared DT 路线”的增强版，不是最终完全体。

师弟，你的行动力太让我惊喜了！我刚刚仔细 Review 了你仓库里最新版本的 `mmdet/models/necks/dt_fpn.py`。

GPT Agent 这一次的修改，可以说是**非常完美的“承上启下”之作**。它准确地执行了我们之前定下的 **“多尺度联合预测单张共享 DT Map” (Shared DT v2)** 的战略！

这个版本的代码在保留了极大工程稳定性的同时，彻底修补了“只在最深层预测会导致分辨率不足、边界丢失”的致命漏洞。

## gemini关于v2的解读

下面师兄为你详细拆解当前这一版 `DTFPN` 的全流程网络结构。一旦你把这套逻辑吃透，你写论文的 Method 章节简直就如同探囊取物！

---

### 一、 宏观数据流：从 Backbone 到 FPN 的奇幻漂流

假设我们输入的是一张 $800 \times 800$ 的密集砾石图像。
1. **Backbone 提特征**：图像穿过 Backbone，吐出了 4 个尺度的特征图 $(C_2, C_3, C_4, C_5)$，分辨率分别是 $200\times200, 100\times100, 50\times50, 25\times25$。
2. **入场 FPN (Lateral Conv)**：这四个特征分别过了一个 $1 \times 1$ 的卷积（`lateral_convs`），统一把通道数变成了 256 维。我们把它们叫做 $L_2, L_3, L_4, L_5$。
3. **【核心魔改】解码 DT Map**：网络**没有**急着做 FPN 的自顶向下融合，而是先拿着这四个 $L$ 特征，跑去算了一张极其精细的距离变换图（`dt_map`）！
4. **FPN 软路由融合**：拿着算好的这唯一一张高清 `dt_map`，在 FPN 自顶向下融合时，作为“空间门控（Gate）”来指挥每一层的特征怎么拼接。

---

### 二、 拆解魔法 1：`dt_map` 是怎么预测出来的？(多尺度联合解码)

这是 GPT 改动的**最大亮点**。它废弃了原来只拿最深层 $L_5$ 猜图的做法，而是手搓了一个**极其轻量化的“微型解码器 (Mini-Decoder)”**。

具体实现在 `_decode_shared_dt_map` 函数里，过程极其优美：
1. **起点 ($L_5$)**：首先把最深、最小的 $L_5$ ($25\times25$) 过一个适配器 `dt_lateral_adapters[-1]`。
2. **逐级上采样与拼接 (Top-down Fusion)**：
   * 把 $L_5$ 放大到 $50\times50$。
   * **拼接 (Concat)**：和当前层的 $L_4$ 拼在一起！
   * **融合 (Fuse)**：过一个 `dt_fuse_blocks`（两个 $3\times3$ 卷积）进行通道降维和特征融合。

# 4_21 Shared DT-FPN v2 复盘与 v3 立项

## 先封存 v2

这一版已经不是“临时实验”，而是后续论文消融里必须保留的基准锚点。先把它的核心产物固定下来：

- best checkpoint: `work_dirs_gravel_big/mask_rcnn_gravel_lightmamba_dt_fpn_v2_36e_gravel_big/best_coco_segm_mAP_epoch_18.pth`
- 训练主日志: `work_dirs_gravel_big/mask_rcnn_gravel_lightmamba_dt_fpn_v2_36e_gravel_big/train.log`
- 时间戳日志: `work_dirs_gravel_big/mask_rcnn_gravel_lightmamba_dt_fpn_v2_36e_gravel_big/20260420_171117/20260420_171117.log`
- 验证曲线: `work_dirs_gravel_big/mask_rcnn_gravel_lightmamba_dt_fpn_v2_36e_gravel_big/20260420_171117/vis_data/scalars.json`
- DT 可视化产物: `work_dirs_gravel_big/mask_rcnn_gravel_lightmamba_dt_fpn_v2_36e_gravel_big/dt_analysis/dt_progress_epochs_8_18_28_36/`
   - 已导出: `epoch_08/`、`epoch_18/`、`epoch_28/`、`epoch_36/`

这条线的定位已经很明确：

- 它证明了 shared DT gate 能显著加快密集边界场景下的前期收敛。
- 它给出了当前最高峰值 `segm mAP = 0.286 @ epoch 18`。
- 它同时暴露了 shared DT 方案的后期结构性瓶颈，为 v3 立项提供了最干净的消融证据。

## v2 已经证明了什么

从这次完整 36e 训练来看，v2 的结论非常清楚：

1. `shared DT map -> shared gate` 不是单纯辅助监督，而是会真实改变 FPN 融合路径。
2. 这套机制对小图密集砾石场景的早期收敛非常有效，`epoch 10` 已经到 `0.278`，`epoch 18` 直接到 `0.286`。
3. v2 的价值不是“最后稳住了多少”，而是它明确展示了：共享几何先验能快速把网络拉到更高上限附近。

也就是说，v2 已经完成了自己的科研任务：

- 正面证明 shared DT 路线有效。
- 反向暴露 shared 设计的尺度冲突问题。

## v2 的核心缺点

### 1. 单张 `dt_shared` 同时服务所有 FPN 融合层，存在天然尺度冲突

当前 v2 的逻辑是：先解码出一张共享高分辨率 `dt_shared`，再把它 resize 到各层去生成 gate。

但 `P2`、`P3`、`P4` 对几何先验的需求并不相同：

- 浅层更需要精细边界与邻近实例分离信息。
- 深层更需要区域结构与语义主体信息。

当所有层都被同一张 `dt_shared` 约束时，本质上就是让不同感受野去服从同一个空间解释，这会在训练后期形成明显的尺度冲突。

### 2. v2 的 Mini-Decoder 仍然把多尺度特征压成了单一几何目标

v2 虽然已经比 v1 好很多，但它仍然在做一件事：

- 把 `L5, L4, L3, L2` 的信息逐级解码成一张唯一的高清 DT 图。

这会带来一个结构性副作用：

- 深层语义和浅层边界都要为同一个 `dt_shared` 服务。
- 多尺度信息不是“各司其职”，而是被迫在同一个监督目标上纠缠。

这解释了为什么 v2 前期冲得很快，但后期会进入“共享约束开始反噬”的阶段。

### 3. 当前 gate 只有 per-level 的标量校准，没有 per-level 的几何内容

现在每层只有：

$$
g_l = \sigma(\alpha_l \cdot \hat d_l + \beta_l)
$$

这里每层能学到的只有一个缩放系数 `alpha_l` 和一个偏置 `beta_l`，但真正的几何内容 `\hat d_l` 仍然来自同一张共享 DT。

这意味着：

- 每层只能“调同一张图的力度”，
- 不能“拥有自己专属的几何判断”。

所以 v2 的门控仍然不是彻底的尺度感知路由。

### 4. 现有证据说明问题出在共享约束，而不是 DT 预测崩掉了

这次 run 最关键的证据不是 `0.286` 本身，而是下面这组组合事实：

- `epoch 18` 达到峰值 `0.286`
- `epoch 36` 回落到 `0.278`
- 但 DT 可视化统计并没有同步恶化

在抽样可视化集上：

- `epoch 18` 平均 `MAE ≈ 0.112485`，`MSE ≈ 0.035147`
- `epoch 36` 平均 `MAE ≈ 0.112752`，`MSE ≈ 0.034649`

这基本说明：

- 不是 DT head 后期预测崩掉了，
- 而是 shared guidance 在后期开始对多层融合施加了过强、过统一的尺度约束。

这正是 v3 应该解决的物理瓶颈。

## v3 目标：Scale-aware Per-level DT-FPN

v3 的核心思想非常明确：

- 废弃“多层特征先拼成一张共享高清 DT 图”的 Mini-Decoder。
- 改为“每一层 top-down 融合时，当场预测本层专属的 `DT_l`，再生成本层专属 gate”。

也就是从：

- `multi-scale -> one shared DT -> all levels share gate source`

切换到：

- `each fusion level -> its own DT_l -> its own gate_l`

这才是真正的尺度感知层级距离路由。

## v3 的前向公式

设当前 top-down 融合目标层为 $l$，当前层 lateral 特征为 $L_l$，上一级输出上采样后为 $U(P_{l+1})$。

v3 的单层流程定义为：

1. 先构造本层路由输入：

$$
R_l = [L_l, U(P_{l+1})]
$$

2. 用本层专属小头预测本层 DT 图：

$$
DT_l = \sigma(\text{Head}_l(R_l))
$$

3. 用本层专属 DT 图生成本层 gate：

$$
g_l = \sigma(\alpha_l \cdot DT_l + \beta_l)
$$

4. 用本层 gate 执行本层融合：

$$
P_l = (1-g_l) \cdot L_l + g_l \cdot U(P_{l+1})
$$

这样 `P2`、`P3`、`P4` 就各自拥有自己的 `DT_l` 和 `g_l`，不再共享同一张几何图。

## v3 的代码重构方案

当前 `dt_fpn.py` 里，v2 的 shared decoder 主要由下面三组模块组成：

- `dt_lateral_adapters`
- `dt_fuse_blocks`
- `dt_predictor`

v3 第一版准备这样改：

### 1. 删除 shared decoder

不再保留：

- `dt_lateral_adapters`
- `dt_fuse_blocks`
- `dt_predictor`
- `_decode_shared_dt_map()`

因为 v3 不再需要先得到一张统一的 `dt_shared`。

### 2. 新增 per-level DT heads

新增：

- `per_level_dt_heads = nn.ModuleList([...])`

长度等于 top-down 融合次数，也就是当前 `num_fusions`。

每个 `Head_l` 的输入都是：

- `concat([L_l, U(P_{l+1})], dim=1)`

输入通道数为 `2 * out_channels`，第一版先保持轻量：

- `3x3 conv -> ReLU -> 1x1 conv -> sigmoid`

这样每一层都只负责自己的 DT 预测，不再跨层共享 DT 内容。

### 3. `forward()` 里把 DT 预测嵌入到每一级 top-down 融合现场

v2 当前是：

1. 先一次性预测 `dt_map`
2. 再进入 top-down 循环
3. 每层只做 resize + gate

v3 则改成：

1. 进入 top-down 循环
2. 对当前层的 `L_l` 与 `upsampled` 做 concat
3. 立刻预测 `DT_l`
4. 立刻得到 `gate_l`
5. 立刻完成本层融合

也就是说，DT 预测与 FPN 融合变成同层闭环，而不是共享前置模块。

### 4. `_last_dt_map` 改成 `_last_dt_maps`

当前为了可视化和导出，neck 里保存的是：

- `_last_dt_map`

v3 需要改为：

- `_last_dt_maps: Tuple[Tensor, ...]`

分别记录每一级融合层的 `DT_l`。

后续导出脚本也要同步升级，支持可视化：

- `P4` 专属 DT
- `P3` 专属 DT
- `P2` 专属 DT

### 5. `_compute_dt_loss()` 改成 multi-level loss

当前 v2 的监督只有一项：

$$
L_{dt}^{v2} = \lambda \cdot \text{MSE}(dt_{shared}, dt_{gt})
$$

v3 第一版应改成：

$$
L_{dt}^{v3} = \sum_{l \in guided} \lambda_l \cdot \text{MSE}(DT_l, \text{Resize}(DT_{gt}, size_l))
$$

第一版消融策略保持简单：

- 先让各层 `lambda_l` 相等
- 总权重先保持与 v2 同量级
- 不引入额外 focal/boundary-aware loss

这样对比 v2 时，变量最干净。

## v3 第一版的实验原则

这一版必须克制，不能一口气把所有想法全堆进去。第一版只验证一件事：

- `shared DT prediction` 换成 `per-level independent DT prediction`，是否能解决 v2 的后期回落问题。

因此 v3 第一版建议严格遵守：

1. 先不引入 IBD。
2. 先不改训练 schedule。
3. 先不加更复杂的边界损失。
4. 尽量复用 v2 的训练 recipe，只改 neck 结构本身。

这样如果 v3 后期稳定性明显提升，结论就会非常硬：

- 提升来自尺度感知层级路由，
- 而不是来自训练技巧。

## 当前结论

现在不该去做“怎么把 v2 的 0.286 用训练技巧保住”，而应该顺着已经暴露出来的物理瓶颈往前推：

- v2 已经证明 shared DT 是有效方向；
- v2 也已经证明 shared DT 不是终局；
- v3 的 per-level DT prediction 才是下一步最值得验证的版本。

因此，后续正式进入：

- `Scale-aware Per-level DT-FPN`
- `Per-level DT Prediction + Per-level Independent Gating`
   * 继续把融合后的特征放大到 $100\times100$ 与 $L_3$ 拼接……再放大到 $200\times200$ 与 $L_2$ 拼接。
3. **输出高清 DT Map**：最后在最高分辨率 ($200\times200$) 上，过一个 $1\times1$ 卷积 `dt_predictor` 和 `Sigmoid`，吐出一张连续的、范围在 $[0, 1]$ 的高清距离场图！

**物理意义**：这张图同时拥有了 $L_5$ 提供的“这里是一大块石头”的**深层高级语义**，以及 $L_2$ 提供的“这里的石缝只有两个像素宽”的**极高频空间细节**。它不再是一团马赛克了！

---

### 三、 拆解魔法 2：损失函数怎么算的？(深度监督的约束)

在 `_compute_dt_loss` 函数中：
1. **提取 GT**：在每次 Forward 之前，由外部 Detector 提取出数据样本中的距离变换真值（`dt_target`），也是在 $[0, 1]$ 之间。
2. **对齐分辨率**：把这张 GT 下采样到与我们刚刚预测的高清 `dt_map` ($200\times200$) 一样大。
3. **算 Loss**：直接用最稳的均方误差 **MSE Loss** `F.mse_loss(dt_map, dt_target)`，并乘以权重 `dt_loss_weight`（目前默认 0.2）。
4. **记录与返回**：算好的 Loss 被存在 `_last_dt_loss` 里，通过 `get_auxiliary_losses` 函数暴露给 Detector 统一反向传播。

---

### 四、 拆解魔法 3：`dt_map` 是怎么作用于 FPN 的？(自适应门控机制)

这是 DT-FPN 的灵魂。在这个阶段，我们要进行 FPN 标准的自顶向下融合（比如把上采样的 $P_5$ 融合进 $L_4$ 得到 $P_4$）。

**在代码的 `forward` 循环中：**
1. **自适应下采样**：虽然我们预测出了一张 $200\times200$ 的超清 `dt_map`，但当我们正在融合 $L_4$ ($50\times50$) 时，这张超清图会被 `F.interpolate` 缩小到 $50\times50$（`resized_dt`）。
2. **可学习的独立仿射变换**：
   ```python
   scale = self.gate_scales[target_level]
   bias = self.gate_biases[target_level]
   gate = torch.sigmoid(scale * resized_dt + bias)
   ```
   这一步**简直绝了**！虽然大家都共用同一张 DT 图，但每一层都有自己**独立的可学习参数 (`scale`, `bias`)**。这意味着 $P_4$ 和 $P_2$ 面对同样的距离值，会根据自己的感受野特性产生不同的门控敏感度。
3. **执行软路由 (Soft Routing)**：
   ```python
   laterals[target_level] = laterals[target_level] * (1.0 - gate) + upsampled * gate
   ```
   *   **如果在接缝处** (`gate` 接近 0)：网络极其不信任 `upsampled` (上采样的深层模糊特征)，而是 100% 信任 `laterals[target_level]` (当前层的高频清晰特征)。粘连被完美切断！
   *   **如果在石头中心** (`gate` 接近 1)：网络高度信任深层语义，避免把石头内部的裂纹当成边界切割。

---

### 💡 师兄的点评与下一步改进建议 (What's Next?)

目前的这套代码，作为我们三步走战略的**“第二步：多尺度联合预测单张 DT Map”**，完成度堪称 100%！它干净、严谨，极其适合跑一版 Ablation（消融实验）。

**还有什么能改进的吗？**

如果你跑完这版之后，发现效果很好，还想进一步冲刺极限（也就是我们的**第三步：多层独立预测 Per-level DT Map**），代码的改进方向非常明确：

1. **废弃共享的微型解码器**：不再需要一层层地 `Concat` 生成一张图。
2. **层级原生感知**：在每一次 top-down 融合 `laterals[i-1]` 和 `upsampled` 时，直接把这两个特征拼在一起（当前层细节 + 深层语义），用一个小卷积头当场预测出这一层专属的 `dt_map` 和 `gate`。
3. **多尺度深度监督**：让每一层预测出来的 `dt_map` 都去和对应尺度的 `dt_target` 算 MSE Loss。

**眼下最紧急的任务**：
师弟，不要再改代码了！这份代码已经极其扎实了。立刻把它挂载到你的网络里，把生成的 `distance_transform_map` 塞进 Dataset，开跑！我们急需这一版实验的数据来验证这个迷你解码器带来的威力！

# 4_21 v3a 失败复盘与 v3b 立项

## 先给 v3a 一个清晰结论

`Per-level DT Prediction + Per-level Independent Gating` 这一版已经完成了它的科研任务，但从结果上看，它并没有解决 v2 的结构瓶颈，反而暴露了“完全去 shared 化”会损失全局几何先验的问题。

这条线需要被保留下来，不是因为它成功，而是因为它给出了下一步 v3b 设计最关键的反证。

当前已经确认的实验现象是：

- v2: `best segm mAP = 0.286 @ epoch 18`
- v3a: `best segm mAP = 0.282 @ epoch 27`
- v3a 到 `epoch 36` 仍只有 `0.279`

也就是说，v3a 不仅没有超过 v2，连 baseline 稳定超越都没有做到。

## v3a 为什么失败

### 1. v3a 把全局几何先验拆没了

v2 的 shared DT decoder 有一个很重要的物理意义：

- `L5 -> L4 -> L3 -> L2` 的整条多尺度路径先汇聚出一张高质量 `dt_shared`
- 这张图虽然会引入尺度冲突，但它至少是一张真正拥有全局视野的几何先验图

v3a 把这一步整个删除之后，每一层的 `DT_l` 都只能从：

$$
R_l = [L_l, U(P_{l+1})]
$$

里自己推断几何。

这就意味着：

- `P2` 只看局部高频和一层深层上采样结果
- `P3` 只看自己的局部上下文
- `P4` 只看更粗的一小段局部语义

三层都没有再被一张共享的全局 DT 图约束。

所以 v3a 的问题不是“per-level 不对”，而是“per-level 时把全局 prior 一起删掉了”。

### 2. v3a 的 DT 监督被拆散了，但没有补回共享主干

v2 只有一张 `dt_shared`，整条 DT 监督都集中压在这一个共享 decoder 上。

v3a 则变成：

- `DT_2` 一份 MSE
- `DT_3` 一份 MSE
- `DT_4` 一份 MSE

虽然实现里是做平均，不是简单手工除以 3，但从优化动力学上看，本质仍然是：

- 共享几何主干没了
- 监督预算被拆给了 3 个更小、视野更窄的 head

这会导致两件事同时发生：

1. 每个 head 学得更慢
2. 每个 head 学到的几何语义更局部、更不稳定

这也是为什么 v3a 的 `loss_neck_dt` 下降速度明显慢于 v2。

### 3. v3a 的三层 DT 都在逼近同一种 GT 语义，但职责并不相同

v3a 当前的监督方式是：

$$
L_{dt}^{v3a} = \sum_l \lambda_l \cdot \text{MSE}(DT_l, \text{Resize}(DT_{gt}, size_l))
$$

这里最大的隐含问题不是 resize 本身，而是：

- `P2` 真正关心的是细边界与邻近实例切分
- `P4` 真正关心的是粗拓扑与主体区域

但三层都被要求去回归“同一张 GT DT 在不同分辨率上的版本”。

也就是说，三层并没有被允许学成“不同层级的最优几何解释”，而是在分别逼近同一个目标的不同缩放版本。

这会带来典型的层间目标不匹配：

- 语义上过于统一
- 路由上却彼此独立

最终表现出来就是：每层都不够差，但也都不够好。

### 4. v3a 失去的是“跨层一致的门控方向”

v2 虽然 shared，但它至少保证所有 guided level 的 gate 都来自同一张 `dt_shared`。

这意味着：

- `P4 -> P3 -> P2` 的 top-down 路由方向是一致的
- 各层只是通过 `alpha_l, beta_l` 调整响应强弱

v3a 中每层都自己预测 `DT_l`，于是变成：

- 每层都能独立决定“哪里该信深层”
- 但三层的决策并不一定彼此一致

所以它不是更自由，而是更容易出现跨层 gate 的几何分歧。

## v3b 的核心思想：Shared Global Prior + Per-level Refinement

v3b 不应该退回纯 v2，也不应该继续纯 v3a，而应该把两者各自正确的部分拼起来。

v3b 的路线是：

1. **保留 v2 的 shared decoder**
   - 依然从 `L5, L4, L3, L2` 解码出一张高质量 `dt_shared`
   - 这张图继续承担全局几何先验

2. **在每个 guided level 上增加 refinement head**
   - 不再让每层白手起家预测 `DT_l`
   - 而是把本层局部信息和中央先验拼在一起，再输出本层 refined DT

3. **最终 gate 使用 refined DT，而不是直接使用 shared DT**
   - 这样可以保住 shared 的全局一致性
   - 同时补上各层真正需要的局部修正

## v3b 的前向公式

先保留 v2 的 shared decoder：

$$
dt_{shared} = \sigma(\text{SharedDecoder}(L_5, L_4, L_3, L_2))
$$

然后对每个 top-down 融合层 $l$：

1. 先把共享 DT 图 resize 到当前层大小：

$$
\hat d_l = \text{Resize}(dt_{shared}, size_l)
$$

2. 构造本层 refinement 输入：

$$
R_l^{refine} = [L_l, U(P_{l+1}), \hat d_l]
$$

3. 用本层 refinement head 输出 refined DT：

$$
DT_l^{refine} = \sigma(\text{RefineHead}_l(R_l^{refine}))
$$

4. 用 refined DT 生成本层 gate：

$$
g_l = \sigma(\alpha_l \cdot DT_l^{refine} + \beta_l)
$$

5. 再执行融合：

$$
P_l = (1-g_l) \cdot L_l + g_l \cdot U(P_{l+1})
$$

这条路线的物理意义非常明确：

- `dt_shared` 负责回答“全局上哪里像中心、哪里像边界”
- `RefineHead_l` 负责回答“在当前层这个局部上下文里，这张全局图应该怎么修”

所以 v3b 不再是“删共享”，而是“先共享，再精修”。

## v3b 为什么比 v3a 更合理

### 1. 它保住了全局视野

v3b 不再让 per-level head 从零开始猜几何，而是先拿到一张已经聚合了 `L5~L2` 的 `dt_shared`。

这能直接修正 v3a 的“信息贫困”问题。

### 2. 它避免了监督彻底碎裂

v3b 不是把监督只拆给 3 个 per-level head，而是同时保留：

- shared branch 的主监督
- refined branch 的 per-level 监督

工程上可以写成两支 loss 的平均：

$$
L_{dt}^{v3b} = \frac{1}{2}L_{shared} + \frac{1}{2}L_{refine}
$$

其中：

$$
L_{shared} = \lambda_{dt} \cdot \text{MSE}(dt_{shared}, dt_{gt})
$$

$$
L_{refine} = \lambda_{dt} \cdot \frac{1}{|G|}\sum_{l \in G} \text{MSE}(DT_l^{refine}, \text{Resize}(dt_{gt}, size_l))
$$

这样 shared prior 不会被放空，per-level refinement 也不会完全无监督。

### 3. 它把 per-level 的自由度限制在“修正”而不是“重建”

v3a 的 per-level head 实际上承担的是“从零重建一张 DT 图”的任务。

v3b 的 refinement head 承担的是：

- 在共享几何先验的基础上做本层修补

这是一个难度更低、也更符合 FPN 分层职责的任务。

### 4. 它同时具备一致性和适配性

v2 的问题是只有一致性，没有层级适配。

v3a 的问题是只有层级适配，没有全局一致性。

v3b 的目标就是把两者拼起来：

- `dt_shared` 提供一致的中心-边界基调
- `DT_l^{refine}` 提供本层专属的局部校正

## v3b 的代码实现路径

### 1. `dt_fpn.py` 中新增第三种模式

当前 neck 已经支持：

- `dt_mode='shared'`
- `dt_mode='per_level'`

v3b 直接新增：

- `dt_mode='shared_refined'`

这样可以最大限度复用 v2/v3a 代码路径，而不必再新造一个 neck 类。

### 2. 保留 v2 的 shared decoder 组件

继续复用：

- `dt_lateral_adapters`
- `dt_fuse_blocks`
- `dt_predictor`
- `_decode_shared_dt_map()`

这部分不删。

### 3. 为 v3b 新增 per-level refinement heads

对每个 guided level 新增一个轻量 head：

- 输入通道：`2 * out_channels + 1`
  - `L_l`
  - `U(P_{l+1})`
  - `Resize(dt_shared)`
- 结构保持轻量：
  - `3x3 conv -> ReLU -> 3x3 conv -> ReLU -> 1x1 conv`

最后经 `sigmoid` 输出本层 refined DT。

### 4. `forward()` 中的执行顺序

v3b 的执行顺序应该是：

1. 先完成 lateral conv
2. 先解码 `dt_shared`
3. 再进入 top-down 循环
4. 每一层取：
   - 当前 `laterals[target_level]`
   - `upsampled`
   - `Resize(dt_shared)`
5. 拼接后通过 `RefineHead_l`
6. 得到 `DT_l^{refine}`
7. 用 `DT_l^{refine}` 生成 gate
8. 完成本层融合

### 5. 可视化接口继续沿用 v3a

为了兼容当前导出工具：

- `get_last_dt_map()` 继续返回 `dt_shared`
- `get_last_dt_maps()` 返回 `P4/P3/P2` 或 `P2/P3/P4` 对应的 refined DT maps（按 guided levels 顺序）

这样 `export_gravel_dt_progress.py` 不用推翻重写，watch 脚本也能直接复用。

## v3b 第一版的实验原则

这一版依然要克制，只验证一个问题：

- 当 shared global prior 被保留，而 per-level 只负责 refinement 时，能否同时保住 v2 的早期上升速度，并缓解 v2 的后期回落。

因此第一版建议仍然坚持：

1. 不引入 IBD
2. 不改 schedule
3. 不上更复杂边界损失
4. 尽量只改 neck 内部结构

## 当前结论

v3a 的失败不是因为“per-level refinement 这条路错了”，而是因为它走成了“per-level from scratch”。

所以 v3b 的正确方向不是回退，也不是继续纯独立分支，而是：

- **保留中央 shared DT 作为全局几何先验**
- **在每个 guided level 上引入 lightweight refinement**
- **让 refined DT 而不是 raw shared DT 去控制本层 gate**

这才是当前最合理、也最值得继续推进的版本。
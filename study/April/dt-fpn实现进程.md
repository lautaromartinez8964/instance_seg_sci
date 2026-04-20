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
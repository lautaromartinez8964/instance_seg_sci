## 4_22 NWD 实现进程

先说明一件事：这个文件名现在叫 `rwd实现进程.md`，但我们当前真正实现和实验的内容是 **NWD（Normalized Gaussian Wasserstein Distance）**。这里先不改文件名，避免把你当前笔记体系打乱；手册正文统一按 NWD 来写。

这条线的定位要先说清楚。

它不是来替代 DT-FPN 的，也不是再做一个“更复杂的 bbox loss 花活”。
它当前最核心的科研任务只有一个：

- **解决小目标在 assignment 阶段因为 IoU 退化而拿不到正样本的问题。**

所以这条线的叙事应该一直保持稳定：

- **DT-FPN** 解决的是 feature path 上的小目标几何信息保真；
- **NWD** 解决的是 training signal 上的小目标正样本分配不足。

这两个模块不是重复关系，而是上下游互补关系。

---

## 一、为什么要做 NWD

### 1. IoU 对 tiny object 很容易退化成 0

对大目标来说，预测框和 GT 框只要中心有一点偏移，IoU 往往还不会立刻掉光。

但对密集小砾石不是这样。

假设一个目标本身只有十几像素大小，哪怕只是 1 到 2 个像素的中心偏差，都可能直接让 anchor 和 GT 变成：

- 几乎不重叠
- 甚至完全不重叠

这时 IoU 会直接掉到 0。

一旦 IoU 变成 0，`MaxIoUAssigner` 的正样本判定就会开始失效：

- 没有 anchor 能达到正样本阈值
- 正样本数量骤减
- RPN recall 上不去
- 后续 RoI head 根本见不到足够多的小目标候选框

这不是分类能力不够，也不是 mask head 不够强，而是训练信号在最前面就断掉了。

### 2. NWD 的核心思想

NWD 把 bbox 不再当作“硬边界矩形”，而是当作一个二维高斯分布。

对于一个框：

$$
b = (x_1, y_1, x_2, y_2)
$$

先把它转成高斯参数：

$$
\mu = \left(\frac{x_1 + x_2}{2}, \frac{y_1 + y_2}{2}\right)
$$

$$
\sigma = \left(\frac{x_2 - x_1}{4}, \frac{y_2 - y_1}{4}\right)
$$

然后计算两个框之间的 Gaussian Wasserstein Distance：

$$
W^2 = \|\mu_1 - \mu_2\|_2^2 + \|\sigma_1 - \sigma_2\|_2^2
$$

最后把距离转成相似度：

$$
	ext{NWD} = \exp\left(-\frac{\sqrt{W^2 + \epsilon}}{C}\right)
$$

其中 $C$ 是归一化常数。

它的物理直觉非常简单：

- IoU 是“有没有碰上”的硬判据；
- NWD 是“离得有多近、尺度差多少”的连续判据。

所以即使两个小框没有重叠，NWD 也不一定是 0。

这就是它最适合 tiny object assignment 的原因。

---

## 二、NWD 在我们这条线里到底想解决什么

你这套 gravel 任务当前已经有一个很清楚的结论：

- DT-FPN v2 把 feature 路由做好了，早期收敛和边界建模已经证明有价值；
- 但小目标 miss 仍然很多，瓶颈主要在 recall，而不是 precision；
- recall 问题里最先该看的，是 proposal 和 assignment，而不是再堆一个新 loss。

所以 NWD 这条线当前最合理的目标，不是“全面替代 IoU”，而是先回答下面三个问题：

1. **只改 RPN assigner，proposal recall 会不会起来？**
2. **如果 RPN 有收益，再把 RCNN assigner 也改成 NWD，会不会继续涨？**
3. **只有前两步都证明 assigner 路线有效，才有必要讨论 bbox loss 要不要跟着一起换。**

这就是为什么当前最推荐的实验顺序必须是：

1. `rpn_only_assign`
2. `rpn_rcnn_assign`
3. `nwd_loss`（可选，且必须最后）

---

## 三、当前代码已经实现了什么

当前 NWD 相关代码主要在下面几个文件里：

- `mmdet/models/task_modules/assigners/nwd_calculator.py`
- `mmdet/models/losses/nwd_loss.py`
- `configs/gravel_big/mask_rcnn_gravel_lightmamba_dt_fpn_nwd_rpn_36e_gravel_big.py`
- `configs/gravel_big/mask_rcnn_gravel_lightmamba_dt_fpn_nwd_rpn_rcnn_36e_gravel_big.py`

它们分别承担不同角色。

### 1. `nwd_calculator.py` 做了什么

这个文件实现的是 **NWD overlap calculator**，也就是给 `MaxIoUAssigner` 提供一个新的“相似度矩阵”来源。

关键函数有三个：

- `bbox_to_gaussian`
- `_compute_nwd_matrix`
- `_compute_nwd_aligned`

它的执行顺序是：

1. 把 `xyxy` 框转成 `(mu, sigma)`
2. 算中心距离和尺度距离
3. 得到 Wasserstein 距离
4. 再指数映射成 `(0, 1]` 的相似度

最后这个相似度矩阵会被 `MaxIoUAssigner` 当成原来 IoU 矩阵的替代物来使用。

也就是说，我们并没有重写 assigner 本身，而是复用了 MMDetection 原有的分配逻辑，只替换了“相似度来源”。

这条实现路径非常干净。

### 2. `nwd_loss.py` 做了什么

这个文件实现的是 **NWDLoss**，也就是把 bbox regression loss 从坐标差异，换成高斯 Wasserstein 相似度损失：

$$
L = 1 - \text{NWD}
$$

它当前的基本思路没有问题：

- 先把预测框和 GT 框转成高斯
- 算 Wasserstein 距离
- 再变成相似度
- 最后用 `1 - similarity` 作为回归损失

但是这条线现在**不应该优先上**，原因很明确：

1. 你当前已经有 DT-FPN 的 `dt_loss`，neck 侧几何监督已经存在；
2. 你现在真正要验证的是 assignment 改造是否能单独提升 recall；
3. 再把 bbox loss 也换掉，会让实验变量一下子变脏；
4. 当前 `NWDLoss` 虽然写出来了，但仍然更适合作为后续可选增强，而不是第一阶段主线。

所以：

- **当前代码里有 NWDLoss 实现**
- **但当前主实验路线不使用它**

这个态度要写死，不要摇摆。

---

## 四、三条实现路线分别是什么

下面把当前三种方法的目的、改动范围和实现方式完全拆开。

### A. `rpn_only` 路线

这是当前第一优先级，也是下一条必须先跑的线。

#### 1. 目标

只测试一件事：

- **把 RPN 的正样本分配从 IoU 换成 NWD 后，小目标 proposal recall 能不能改善。**

#### 2. 改动位置

配置文件：

- `configs/gravel_big/mask_rcnn_gravel_lightmamba_dt_fpn_nwd_rpn_36e_gravel_big.py`

它继承：

- `mask_rcnn_gravel_lightmamba_dt_fpn_v2_36e_gravel_big.py`

也就是说，这条线默认保留：

- DT-FPN v2 neck
- DT loss
- 原来的 RPN/RCNN bbox loss
- 原来的 RoI assigner

只改一个地方：

- `model.train_cfg.rpn.assigner.iou_calculator`

从：

- `BboxOverlaps2D`

改成：

- `BboxOverlaps2D_NWD`

#### 3. 当前实现逻辑

这份配置做的事情是：

- RPN assigner 里把 overlap calculator 换成 NWD
- `pos_iou_thr=0.7`
- `neg_iou_thr=0.3`
- `min_pos_iou=0.3`

注意，虽然字段名还叫 `iou_thr`，但语义已经变成“对 NWD 相似度做阈值切分”。

这是一个工程兼容写法，不代表它仍然真的是 IoU。

#### 4. 这条线的价值

它能最干净地回答：

- proposal recall 的增益是否主要来自 RPN 这一层的正样本恢复。

如果这条线都没信号，那就不该急着把 RCNN 和 loss 一起堆上去。

---

### B. `rpn_rcnn` 路线

这是第二阶段路线，不该抢跑。

#### 1. 目标

在 RPN 已经切到 NWD 的基础上，再把 RoI head 的 assigner 也改成 NWD，看二阶段匹配是否继续带来收益。

#### 2. 改动位置

配置文件：

- `configs/gravel_big/mask_rcnn_gravel_lightmamba_dt_fpn_nwd_rpn_rcnn_36e_gravel_big.py`

它相当于：

- RPN assigner -> NWD
- RCNN assigner -> NWD
- bbox regression loss 仍保持原版

#### 3. 当前实现逻辑

这份配置会改两个地方：

1. `train_cfg.rpn.assigner.iou_calculator = BboxOverlaps2D_NWD`
2. `train_cfg.rcnn.assigner.iou_calculator = BboxOverlaps2D_NWD`

其中 RCNN 仍沿用原本的阈值：

- `pos_iou_thr=0.5`
- `neg_iou_thr=0.5`
- `min_pos_iou=0.5`

#### 4. 这条线的意义

它回答的是：

- proposal 已经改善以后，RoI 阶段的正负样本匹配是否也值得同步切到 NWD。

这一步只有在 `rpn_only` 已经证明方向成立之后才值得做。

否则你会遇到一个很糟糕的问题：

- 指标涨了，不知道是 RPN 起作用，还是 RCNN 起作用；
- 指标跌了，也不知道是 RPN 阈值没调好，还是 RCNN 把噪声 proposal 放大了。

所以它必须排在第二步。

---

### C. `nwd_loss` 路线

这是第三阶段路线，而且当前大概率不是最终版必须项。

#### 1. 目标

把 bbox regression 的优化目标也切到 NWD，让训练的“匹配准则”和“回归准则”统一起来。

#### 2. 代码位置

核心实现在：

- `mmdet/models/losses/nwd_loss.py`

理论上要真正接入，需要额外处理：

- `reg_decoded_bbox=True`
- 或者在 loss 内部显式 decode

这是因为 NWD 作用对象必须是 **绝对坐标框**，不能直接拿编码后的 `dx, dy, dw, dh` 去算 Wasserstein 距离。

#### 3. 为什么它当前不该优先上

原因非常清楚：

1. 你现在已经有 `dt_loss`，neck 侧几何监督并不缺；
2. 你现在真正缺的是 assigner 侧的小目标正样本恢复；
3. 如果现在就把 loss 也换了，变量会立刻从“匹配机制实验”变成“匹配 + 回归联合实验”；
4. 论文归因会变得很难看。

#### 4. 当前建议结论

对当前项目来说：

- **NWD loss 可以存在，但不应该作为当前主线优先项。**

甚至最终版里不一定需要它。

如果最后：

- `rpn_only` 有稳定收益
- `rpn_rcnn` 继续有增益
- 但仍然存在 bbox regression 明显拉胯

那时再考虑它，逻辑才干净。

---

## 五、为什么现在推荐“先 assigner，后 loss”

这部分一定要写清楚，因为它直接决定后续论文实验顺序。

### 1. 当前最该验证的是 recall 源头，而不是回归形式

现在我们不是不知道 bbox loss 怎么写，而是不知道：

- 小目标 miss 的主要矛盾，到底是不是来自 assignment。

所以最先该做的，是让实验回答这个问题。

### 2. DT-FPN 已经承担了一部分几何监督角色

当前 baseline 不是普通 FPN，而是 DT-FPN v2。

这意味着你已经有：

- neck 内部 DT 预测
- 显式 `dt_loss`
- 几何先验驱动的 top-down 融合

在这种前提下，再一上来就加 NWD loss，很容易让方法显得职责重叠。

最合理的模块分工应该是：

- DT-FPN 管 feature routing
- NWD 管 sample assignment
- 原生 bbox loss 先不动

这样最清楚，也最容易写论文。

### 3. 工程风险最小

只改 assigner 的优势是：

- 不改 head 输出维度
- 不改 bbox coder
- 不改 decode 流程
- 不碰 regression target 表达形式

所以它既是科研上最干净的第一步，也是工程上最稳的第一步。

---

## 六、当前推荐实验顺序

这部分直接作为后续执行标准。

### 第一阶段：`nwd_rpn_only`

先跑：

- `configs/gravel_big/mask_rcnn_gravel_lightmamba_dt_fpn_nwd_rpn_36e_gravel_big.py`

目的：

- 验证 NWD 是否能在 **RPN proposal assignment** 这一层给出有效提升。

这一阶段重点观察：

- 整体 `segm mAP`
- `mAP_s`
- `AP50`
- 训练前中期收敛速度
- proposal recall 相关现象

### 第二阶段：`nwd_rpn_rcnn`

只有当第一阶段已经有正信号，才跑：

- `configs/gravel_big/mask_rcnn_gravel_lightmamba_dt_fpn_nwd_rpn_rcnn_36e_gravel_big.py`

目的：

- 验证把 RoI assigner 也切到 NWD 后，收益是否还能继续扩大。

### 第三阶段：`nwd_loss`

只有前两者已经证明 assignment 路线成立，才考虑这一阶段。

这一阶段不是默认任务，而是可选增强。

如果上，必须先保证：

- loss 输入是真实 decoded bbox
- 接口和 MMDetection 调用约定完全对齐
- 实验归因仍然可解释

---

## 七、当前阶段的最终判断

把这条线一句话说透就是：

- **当前 NWD 的主任务不是替换 loss，而是先把小目标 assignment 做对。**

所以在现阶段：

1. 先启动 `nwd_rpn_only`，这是当前第一优先级；
2. 如果它有效，再做 `nwd_rpn_rcnn`；
3. `nwd_loss` 只保留为第三阶段候选项，不作为当前主线刚需。

这就是当前最稳、最干净、也最适合写进论文方法与消融逻辑的一条路线。

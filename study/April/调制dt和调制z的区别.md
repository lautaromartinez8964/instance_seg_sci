

好，这个问题非常核心。我从 SSM 的数学本质出发，把整个链条拆干净。

---

## 第一部分：你的 importance 机制在遥感密集目标实例分割里到底能不能起作用

### 1.1 先看 importance 到底提供了什么信息

你的 `ForegroundHead` 的输出是一个像素级的概率图 `importance ∈ (B, 1, H, W)`，值域 [0, 1]，由 BCE loss 监督，目标是"所有实例 mask 的 union"。

```python name=mmdet/models/backbones/rs_lightmamba/fg_ig_scan.py url=https://github.com/lautaromartinez8964/instance_seg_sci/blob/822be854ce968ac851b3319f289b1ced537ec4cc/mmdet/models/backbones/rs_lightmamba/fg_ig_scan.py#L55-L60
def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.act(self.bn1(self.pw_down(x)))
    x = self.act(self.bn2(self.dw_large(x)))
    x = self.act(self.bn3(self.pw_mix(x)))
    x = self.act(self.bn4(self.dw_refine(x)))
    return torch.sigmoid(self.pw_out(x))
```

它本质上是一个**二分类语义分割头**——"这个位置是不是前景"。

v3 的训练已经证明了：**在有 BCE 辅助监督的条件下，这个头能准确区分前景和背景。** 密集车辆区域呈现高响应，空旷背景呈现低响应。

### 1.2 遥感实例分割为什么需要这个信息

iSAID 数据集的统计特征：

| 属性 | 数值 |
|---|---|
| 前景面积占比 | 5-15% |
| 背景面积占比 | 85-95% |
| 单张图内实例数量 | 几十到几百（密集停车场） |
| 小目标面积 | 150-300 px²（small vehicle） |
| 目标分布 | 极度空间不均匀——大片空地 + 局部密集簇 |

VMamba 的标准四方向扫描对每个 token 一视同仁。SSM 的隐状态维度是固定的（`d_state=16`），它必须用这 16 维去编码序列里所有 token 的信息。当 85% 的 token 是背景时，**隐状态的"容量"被背景信息填满，前景信息被挤压**。

这不是理论推测，而是 SSM 递推的数学必然：

```
h_t = Ā · h_{t-1} + B̄ · x_t

假设连续 100 个背景 token 后遇到 1 个前景 token：
h_100 ≈ Ā^100 · h_0 + Σ(Ā^k · B̄ · x_bg_k)
                        └──────────────────┘
                        隐状态被 100 个背景 token 的信息填满

当第 101 个 token 是前景时：
h_101 = Ā · h_100 + B̄ · x_fg
        └── 充满背景信息的状态 ──┘

x_fg 的信息只能"加"进去，无法"替换"掉 h_100 里的背景信息
除非 dt 足够大让 Ā 衰减得很快——但 dt 是从 x 的特征隐式学的，
网络在 85% 背景的训练数据中，dt 的分布自然偏向"对背景友好"
```

**所以 importance 提供的信息——"这个位置是前景还是背景"——在遥感场景下确实是 SSM 迫切需要但自身难以充分学到的先验。**

### 1.3 为什么说"SSM 自己难以学到"

你可能会问：Mamba 不是 **selective** state space model 吗？B、C、dt 都是数据相关的，它不是应该自己学会"关注前景"吗？

理论上是的。但在实践中有两个障碍：

**第一，监督信号太间接。** SSM 的 B/C/dt 参数从来没有被显式告知"哪里是前景"。它们只能从最终的 detection/segmentation loss 经过长链反传来隐式学习。这条梯度路径是：`mask head loss → FPN → backbone stage 4 → SSM output → C·h → h → Ā → dt`，链条极长，信号极弱。

**第二，训练数据的分布偏差。** 85% 的训练信号来自背景 token。在随机梯度下降的统计意义上，dt/B/C 的参数更新被背景 token 的梯度主导。网络确实能学到一些前景/背景差异，但远不如你用一个显式的 BCE loss 直接监督来得准确和稳定。

### 1.4 结论

**importance 机制能不能起作用？能。** 条件是：

1. importance head 被正确监督（v3 已证明）
2. importance 信号被注入到 SSM 能"感知到"的位置（v3 的重排方式不合理，z-gate 和 dt 方式更合理）
3. 注入方式不破坏 SSM 的空间连续性假设

---

## 第二部分：VMamba 原版的 SS2D 到底是怎么工作的

在分析两个方案之前，必须把原版的数据流完全拆清楚。我直接从你仓库里的源码来讲。

### 2.1 SS2D 的完整 forward 路径

```python name=mmdet/models/backbones/vmamba_official/vmamba.py url=https://github.com/lautaromartinez8964/instance_seg_sci/blob/822be854ce968ac851b3319f289b1ced537ec4cc/mmdet/models/backbones/vmamba_official/vmamba.py#L663-L679
def forwardv2(self, x: torch.Tensor, **kwargs):
    x = self.in_proj(x)                          # (B,H,W,C) → (B,H,W, d_proj)
    if not self.disable_z:
        x, z = x.chunk(2, dim=-1)                # x: (B,H,W,d_inner), z: (B,H,W,d_inner)
        if not self.disable_z_act:
            z = self.act(z)                       # z = SiLU(z)
    x = x.permute(0,3,1,2).contiguous()          # → (B, d_inner, H, W)
    if self.with_dconv:
        x = self.conv2d(x)                        # depthwise 3×3 conv
    x = self.act(x)                               # SiLU(x)
    y = self.forward_core(x)                      # ← 核心：CrossScan → SSM → CrossMerge
    y = self.out_act(y)
    if not self.disable_z:
        y = y * z                                 # ← 门控乘法
    out = self.dropout(self.out_proj(y))
    return out
```

画成图：

```
输入 x_in (B, H, W, C)
    │
    in_proj   →  d_proj = 2 × d_inner (有z) 或 d_inner (无z)
    │
    ├─── x_main (B, H, W, d_inner) ──→ permute → conv2d → SiLU
    │                                              │
    │                                    ┌─────────┴─────────┐
    │                                    │   forward_core     │
    │                                    │                    │
    │                                    │  CrossScan(4方向)   │
    │                                    │      ↓             │
    │                                    │  x_proj → dts,B,C  │
    │                                    │      ↓             │
    │                                    │  selective_scan_fn  │
    │                                    │  (SSM 递推)         │
    │                                    │      ↓             │
    │                                    │  CrossMerge(求和)   │
    │                                    │      ↓             │
    │                                    │  out_norm          │
    │                                    └─────────┬─────────┘
    │                                              │
    │                                         y_ssm (B, d_inner, H, W)
    │                                              │
    └─── z_gate (B, H, W, d_inner) → SiLU(z) ────×──→ y = y_ssm × z
                                                   │
                                              out_proj → dropout → 输出
```

### 2.2 forward_core 内部——SSM 递推的核心

```python name=vmamba_forward_core url=https://github.com/lautaromartinez8964/instance_seg_sci/blob/822be854ce968ac851b3319f289b1ced537ec4cc/mmdet/models/backbones/vmamba_official/vmamba.py#L540-L620
def forward_corev2(self, x, ...):
    # x: (B, d_inner, H, W) — 经过 conv2d + SiLU 之后

    # 1. CrossScan: 将 2D 特征展开为 4 条 1D 序列
    #    行正序、行逆序、列正序、列逆序
    xs = CrossScan.apply(x)  # (B, 4, d_inner, H×W)

    # 2. 投影得到 SSM 参数
    x_dbl = conv1d(xs, x_proj_weight)  # → dts_raw, Bs, Cs
    dts, Bs, Cs = split(x_dbl, [dt_rank, d_state, d_state])
    dts = conv1d(dts, dt_projs_weight)  # dt_rank → d_inner

    # 3. SSM 递推（4条路径同时执行）
    ys = selective_scan_fn(xs, dts, As, Bs, Cs, Ds, delta_bias, ...)
    # ys: (B, 4, d_inner, H×W)

    # 4. CrossMerge: 4条路径的输出逆排列后求和
    y = CrossMerge.apply(ys)  # (B, d_inner, H, W)
    y = out_norm(y)
    return y
```

### 2.3 selective_scan_fn 内部的逐 token 递推

这是最核心的部分。对于序列中的每个 token t：

```
输入：x_t (d_inner维), dt_t (d_inner维), B_t (d_state维), C_t (d_state维)
参数：A (d_inner × d_state), D (d_inner)

Step 1: 离散化
    dt_t' = softplus(dt_t + delta_bias)     ← dt 经过 softplus 变为正值
    Ā_t = exp(A × dt_t')                    ← 状态转移矩阵（对角矩阵）
    B̄_t = dt_t' × B_t                       ← 输入矩阵（简化近似）

Step 2: 状态递推
    h_t = Ā_t ⊙ h_{t-1} + B̄_t ⊗ x_t       ← h_t: (d_inner, d_state)
    ⊙: 逐元素乘（沿 d_state 维）
    ⊗: 外积扩展（x_t 的 d_inner 维 × B̄_t 的 d_state 维）

Step 3: 输出
    y_t = (C_t · h_t) + D × x_t             ← y_t: (d_inner维)
    C_t · h_t: 沿 d_state 维点积
```

**关键理解：dt_t 同时控制两件事：**

```
Ā_t = exp(A × dt_t')

A 是负数（初始化为 -exp(A_log)），所以 A × dt_t' 是负数。

dt_t' 越大：
  → A × dt_t' 越负
  → Ā_t = exp(更负的数) → 更接近 0
  → h_{t-1} 被更多"遗忘"
  
同时：
  → B̄_t = dt_t' × B_t 也变大
  → x_t 被更多"写入"隐状态

总结：dt_t 大 = "忘掉旧的，写入新的"
     dt_t 小 = "保留旧的，忽略新的"
```

---

## 第三部分：z-gate 调制方案——机制解剖

### 3.1 原版 VMamba 中 z 做了什么

z 分支从 `in_proj` 分出来，**完全不经过 SSM**，直接经过 SiLU 后和 SSM 输出做逐元素乘法：

```python name=vmamba.py url=https://github.com/lautaromartinez8964/instance_seg_sci/blob/822be854ce968ac851b3319f289b1ced537ec4cc/mmdet/models/backbones/vmamba_official/vmamba.py#L665-L677
x, z = x.chunk(2, dim=-1)        # z 和 x 来自同一个 in_proj 的两半
z = self.act(z)                    # SiLU(z)
...
y = self.forward_core(x)          # x 经过 SSM
y = self.out_act(y)
y = y * z                          # ← y 和 z 逐元素相乘
```

z 的作用是 **GLU 门控**（Gated Linear Unit）。这个设计来自 Mamba 论文原始架构，继承自 GPT-style 的 FFN 中 `FFN_SwiGLU(x) = (xW₁) ⊙ SiLU(xW₂)`。

**z 门控的本质含义：**

```
y_ssm 包含了 SSM 递推得到的"序列建模特征"
z 包含了 in_proj 直接投影的"原始输入特征"

y = y_ssm ⊙ SiLU(z)

这等价于：
  z 决定了 y_ssm 的每个元素被"放行"多少
  z 大的位置/通道：SSM 输出被充分利用
  z 小的位置/通道：SSM 输出被抑制
```

**在原版 VMamba v05_noz 中，z 被去掉了：**

```python name=vmamba.py url=https://github.com/lautaromartinez8964/instance_seg_sci/blob/822be854ce968ac851b3319f289b1ced537ec4cc/mmdet/models/backbones/vmamba_official/vmamba.py#L453-L454
# in proj =======================================
d_proj = self.d_inner if self.disable_z else (self.d_inner * 2)
```

`disable_z=True` 时，`in_proj` 只投影到 `d_inner`，没有 z 分支，`y` 直接等于 `y_ssm`。

### 3.2 你的 v4 z-gate 调制做了什么

你的代码在恢复 z 分支的基础上，让 importance 调制 z 的空间分布：

```python name=mmdet/models/backbones/rs_lightmamba/ig_ss2d.py url=https://github.com/lautaromartinez8964/instance_seg_sci/blob/822be854ce968ac851b3319f289b1ced537ec4cc/mmdet/models/backbones/rs_lightmamba/ig_ss2d.py#L143-L168
def forwardv2(self, x: torch.Tensor, **kwargs):
    x = self.in_proj(x)
    if not self.disable_z:
        x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
        if not self.disable_z_act:
            z = self.act(z)                                    # SiLU(z)
    ...
    importance = self.ig_scan_module.predict_importance(x)     # (B,1,H,W)
    y = self.forward_core(x)                                   # 标准 SSM
    y = self.out_act(y)
    if not self.disable_z:
        if self.ig_mode == 'z_gate' and importance is not None:
            z_importance = importance                           # (B,1,H,W)
            z = z * (1.0 + torch.clamp(self.gate_scale, min=0.0) * z_importance)
        y = y * z                                              # 调制后的门控
```

**和原版的精确对比：**

```
原版 VMamba (有z):
    z = SiLU(in_proj(x)的后半部分)
    y = y_ssm × z
    → z 的空间分布完全由 in_proj 的参数决定，没有显式的前景/背景偏好

原版 VMamba v05 (无z):
    y = y_ssm
    → 没有门控，SSM 输出直接送 out_proj

你的 v4 z-gate:
    z = SiLU(in_proj(x)的后半部分)
    z = z × (1 + γ × importance)          ← importance 调制
    y = y_ssm × z
    → z 在前景位置被放大，在背景位置保持不变（当前是单边）
```

### 3.3 z-gate 调制的物理意义

把 `y = y_ssm × z` 展开到每个空间位置 (h, w)：

```
对于位置 (h, w) 的第 c 个通道：

原版：
    y[h,w,c] = y_ssm[h,w,c] × SiLU(z[h,w,c])

v4 z-gate：
    y[h,w,c] = y_ssm[h,w,c] × SiLU(z[h,w,c]) × (1 + γ × importance[h,w])
```

**这实际上在做什么：**

```
前景位置 (importance ≈ 1):
    y[h,w,c] = y_ssm[h,w,c] × SiLU(z[h,w,c]) × (1 + γ)
    → SSM 输出被放大 (1+γ) 倍
    → 这个位置的特征在后续 FPN/detection head 中信号更强

背景位置 (importance ≈ 0):
    y[h,w,c] = y_ssm[h,w,c] × SiLU(z[h,w,c]) × 1.0
    → SSM 输出保持原样
```

**z-gate 调制作用在 SSM 流水线的哪个环节：**

```
in_proj → [x_main] → conv2d → SiLU → CrossScan → SSM递推 → CrossMerge → out_norm → y_ssm
                                                                                       │
in_proj → [z_gate] → SiLU → ★ importance 调制 ★ ─────────────────────────────────→ × ──┤
                                                                                       │
                                                                                  out_proj → 输出
```

**作用时机：SSM 递推已经全部完成之后，在输出被送入 `out_proj` 之前。**

### 3.4 z-gate 调制的本质局限

这是一个我需要对你说清楚的事实：

**z-gate 调制不改变 SSM 内部的任何计算。** SSM 递推的 h_t、Ā_t、B̄_t、C_t 没有任何一个感知到了 importance 的存在。z-gate 调制做的事情等价于：

```
y_final = y_ssm × spatial_attention_mask
```

其中 `spatial_attention_mask = SiLU(z) × (1 + γ × importance)`。

**这在数学上就是一个 spatial attention gate。** 它和 CBAM 的 spatial attention、SE-Net 的 channel attention、DAMamba 的 center_feature_scale，在抽象层面上是同一类操作——都是在某一层特征上乘一个数据相关的权重 mask。

**差异在于：**
1. 你的 mask 有显式前景 BCE 监督（CBAM/SE 没有）
2. 你的 mask 作用在 SSM 的 z 分支上而不是外挂的 attention 模块（位置不同）
3. 你的场景是遥感前景稀疏（task-specific 的动机）

这些差异足够写论文，但说实话，创新深度有限。

---

## 第四部分：dt 调制方案——机制解剖

### 4.1 dt 在 SSM 递推中的精确位置

回到 selective_scan 的逐 token 计算：

```
对序列中第 t 个 token：

1. dt_raw_t = x_proj(x_t)[:dt_rank]              ← 从输入特征投影得到
2. dt_projected_t = dt_projs_weight × dt_raw_t    ← dt_rank → d_inner
3. dt_t = softplus(dt_projected_t + delta_bias)   ← 保证为正

4. Ā_t = exp(A × dt_t)                            ← 状态转移系数
5. B̄_t = dt_t × B_t                               ← 输入缩放系数
6. h_t = Ā_t ⊙ h_{t-1} + B̄_t ⊗ x_t              ← 状态更新
7. y_t = C_t · h_t + D × x_t                      ← 输出
```

**dt 的调制点应该在第 3 步之后、第 4 步之前。** 也就是在 dt 经过 softplus 成为正值之后，在它参与指数运算之前。

但在你的代码里，由于 softplus 和 delta_bias 是在 `selective_scan_fn` 的 CUDA kernel **内部** 执行的，你无法直接在第 3 步和第 4 步之间插入代码。

所以实际的调制点是在 `dts` 被送入 `selective_scan_fn` **之前**：

```python name=mmdet/models/backbones/rs_lightmamba/ig_ss2d.py url=https://github.com/lautaromartinez8964/instance_seg_sci/blob/822be854ce968ac851b3319f289b1ced537ec4cc/mmdet/models/backbones/rs_lightmamba/ig_ss2d.py#L118-L134
xs = xs.view(batch_size, -1, seq_len)
dts = dts.contiguous().view(batch_size, -1, seq_len)   # ← 调制点在这里之后
As = -self.A_logs.to(torch.float).exp()
Ds = self.Ds.to(torch.float)
...
delta_bias = self.dt_projs_bias.view(-1).to(torch.float)

ys = selective_scan_fn(
    xs, dts, As, Bs, Cs, Ds, delta_bias, True, ssoflex,    # dts 进入 kernel
    backend=selective_scan_backend)
```

在 kernel 内部，实际执行的是 `dt_t = softplus(dts[t] + delta_bias)`。所以你如果在外面乘了 importance：

```
dts_modulated[t] = dts[t] × (1 + β × importance[t])
→ kernel 内部: dt_t = softplus(dts_modulated[t] + delta_bias)
→ 等价于: dt_t = softplus(dts[t] × (1 + β × imp) + delta_bias)
```

**注意：这不完全等价于 `softplus(dts[t] + delta_bias) × (1 + β × imp)`。** 因为 softplus 是非线性函数，乘法在 softplus 之前和之后效果不同。在 softplus 之前乘，实际上是在对数空间里做缩放，效果更平滑：

```
softplus(a × k) ≈ a × k  当 a×k 较大时（因为 softplus(x) ≈ x for x >> 0）
softplus(a × k) ≈ log(1 + exp(a×k))  当 a×k 较小时

所以在 softplus 之前乘 importance：
  前景(imp高): dts × 1.1 → softplus 输出略大 → dt 略大
  背景(imp低): dts × 1.0 → softplus 输出不变 → dt 不变

变化是渐进的、平滑的，不会出现 dt 突变的问题。
```

### 4.2 dt 调制的物理意义——逐步推导

**Step 1：前景位置的 dt 被增大**

```
importance[前景] ≈ 0.8~1.0
dt_modulated = dt_original × (1 + β × 0.9)    假设 β=0.1
             = dt_original × 1.09

dt 从比如 0.05 变成 0.0545
```

**Step 2：更大的 dt 如何影响状态转移**

```
Ā = exp(A × dt)

A ≈ -1 (典型初始化后学到的值)

原始: Ā = exp(-1 × 0.05) = exp(-0.05) ≈ 0.951
调制后: Ā = exp(-1 × 0.0545) = exp(-0.0545) ≈ 0.947

差异: 0.951 → 0.947
```

看起来很小对不对？但注意这是**逐步累积**的：

```
经过 10 个连续的前景 token 后：

原始: Ā^10 = 0.951^10 ≈ 0.606  → 保留了 60.6% 的初始状态
调制后: Ā^10 = 0.947^10 ≈ 0.581  → 保留了 58.1% 的初始状态

差异：2.5%

经过 50 个连续的背景→前景过渡后，这个累积差异会放大。
```

**Step 3：更大的 dt 同时如何影响输入写入**

```
B̄ = dt × B

原始: B̄ = 0.05 × B
调制后: B̄ = 0.0545 × B

→ 前景 token 的输入被放大了 9%，更多的前景信息被写入隐状态
```

**综合效果：**

```
dt 增大（前景位置）→ 同时发生两件事：
  ① Ā 减小 → 遗忘更多历史（更快清除之前的背景信息）
  ② B̄ 增大 → 写入更多当前输入（更充分编码前景信息）

dt 不变（背景位置）→ 保持原始行为
```

**这就是 dt 调制比 z-gate 调制更深刻的根本原因：它改变的是 SSM 的"记忆策略"本身，而不只是输出的缩放。**

### 4.3 dt 调制和 z-gate 调制的本质区别

我用一张图把两者的作用点画清楚：

```
                          ┌──────────────── SSM 递推内部 ────────────────┐
                          │                                              │
x_t ──→ x_proj ──→ dts ──┤──→ ★dt调制★ ──→ softplus ──→ Ā, B̄          │
                          │                              │              │
                          │              h_{t-1} ──→ Ā⊙h + B̄⊗x ──→ h_t │
                          │                                      │      │
                          │                              C_t ──→ C·h ──→│──→ y_ssm_t
                          │                                              │
                          └──────────────────────────────────────────────┘
                                                                    │
                                                                    │
z_t ──→ SiLU ──→ ★z-gate调制★ ──→ y_ssm_t × z_modulated_t ──→ y_final_t
```

| 维度 | z-gate 调制 | dt 调制 |
|---|---|---|
| **作用位置** | SSM **外部**（门控分支） | SSM **内部**（状态递推之前） |
| **改变了什么** | SSM 输出的空间缩放 | SSM 的记忆/遗忘策略 |
| **影响范围** | 只影响当前 token 的输出幅值 | **影响当前 token 及所有后续 token 的隐状态演化** |
| **类比** | 音量旋钮（调大调小） | **耳朵的选择性注意力（决定听不听进去）** |
| **数学影响** | `y *= scale` | `h_t = exp(A·dt·scale)·h_{t-1} + dt·scale·B·x_t` |
| **是否改变 SSM 内部** | ❌ 完全不改 | ✅ 改变 Ā 和 B̄ |
| **创新深度** | 中（spatial attention 变体） | **高（SSM 选择性机制的显式调控）** |

### 4.4 dt 调制为什么对遥感场景特别合适

回到最开始的问题——"连续 100 个背景 token 之后遇到前景"：

```
原版 SSM（无 dt 调制）：
  背景 token 1-100: dt 由 x_proj 隐式决定，大概率不大
  → h_100 充满了背景信息
  前景 token 101: dt 由 x_proj 隐式决定，可能稍大但不确定
  → h_101 = 0.95·h_100 + small_B × x_fg
  → 前景信息被稀释

dt 调制后：
  背景 token 1-100: dt 不变（importance ≈ 0）
  → h_100 充满背景信息（和原版一样）
  前景 token 101: dt 被 importance 增大
  → Ā_101 更小（比如 0.90 而不是 0.95）→ h_100 被更快遗忘
  → B̄_101 更大 → x_fg 被更充分写入
  → h_101 = 0.90·h_100 + larger_B × x_fg
  → 前景信息在隐状态中的占比更高

后续前景 token 102, 103, ...（如果在同一个密集区域）：
  每个都有高 importance → dt 持续偏大 → 隐状态持续被前景信息刷新
  → 形成一个"前景信息主导的隐状态段"
```

**这就是 dt 调制解决"遥感前景状态稀释问题（RS-FSD）"的精确机制。**

---

## 第五部分：两个方案的对比总结

### 5.1 修改幅度对比

**z-gate 调制需要改的地方：**

1. `forward_type` 从 `v05_noz` 改为 `v05`（恢复 z 分支）
2. `in_proj` 维度从 `d_inner` 扩展到 `2×d_inner`
3. 在 `forwardv2` 的 `y * z` 之前插入 importance 调制
4. 需要处理预训练权重不适配（`in_proj` 维度变了）

**dt 调制需要改的地方：**

1. `forward_type` 保持 `v05_noz`（不需要 z 分支）
2. `in_proj` 维度不变
3. 在 `forward_core` 的 `dts` 计算之后、`selective_scan_fn` 之前插入 importance 调制
4. **预训练权重完全兼容，不需要任何维度适配**

### 5.2 风险对比

| 风险 | z-gate | dt |
|---|---|---|
| 预训练不适配 | ⚠️ **严重**（z 半边全随机） | ✅ 无此问题 |
| 审稿人质疑"VMamba 已去掉 z，你为什么加回来" | ⚠️ 需要论证 | ✅ 无此问题 |
| 训练不稳定 | 低 | ⚠️ **中**（β 必须小） |
| 创新度不够 | ⚠️ 本质是 spatial attention | ✅ SSM 内部调制，无先例 |

### 5.3 我的最终判断——综合上一轮讨论

上一轮你和 GPT 讨论后的共识是"先修复 v4-a z-gate 的实现缺陷（BN→GN、单向→双向），给它一次公平验证"。**我仍然同意这个策略。**

但我要补充一个很务实的考量：

**你的时间线紧迫。** 根据你的学习计划，Phase 1B（IG-Scan）的 deadline 是 4.22，你已经花了 v1→v2→v3→v4 四个版本在扫描机制上。如果 z-gate 修复后仍然不行，你就要在 4 月内再切一次方向。

所以我建议一个**并行策略**：

```
本周（4.2-4.5）：
  ① 修复 v4-a z-gate 的三个问题（BN→GN、单向→双向、新参数学习率）
  ② 同时实现一个 dt 调制的最小版本（就是 ~10 行代码的事）

下周（4.6-4.10）：
  ③ v4-b (z-gate 修复版) 跑 12 epoch
  ④ v4-dt (dt 调制版) 跑 12 epoch
  ⑤ 对比两者 vs baseline

这样无论哪个赢，你都有结果可用，不会空耗时间。
```

dt 调制版的实现极其简单，因为**不需要恢复 z 分支、不需要改 `in_proj` 维度、不需要处理预训练适配**——就是在 `forward_core` 的 `dts` 后面加一行乘法。把两个同时跑，一周内出结果，然后选赢家继续推进。
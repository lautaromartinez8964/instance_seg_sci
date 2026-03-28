# 🎯 RS-LightMamba 完整创新手册 v4（2026-03-23 最终版）

> **定位**：论文方法定稿 + 代码实现协议 + 写作模板 + 审稿应对
> **投稿目标**：IEEE TGRS（保底 JSTARS / Remote Sensing）
> **三大模块**：
>   **(A) FG-IG-Scan — 核心架构创新**
>   **(B) Integral Distillation — 核心训练创新**
>   **(C) FGEB — 轻量频域边界增强（第三模块，受 CFG-MambaNet 启发）**
> **AFM 定位**：次要备选，仅在附录做对照（None vs AFM vs FGEB），主文不列为贡献

---

## 0. 为什么本版把 AFM 撤回、换上 FGEB

你在 `创新点_26_3_19_gpt.md` 中的判断是正确的：

| 维度 | AFM | FGEB |
|---|---|---|
| **技术深度** | 空间域三分支加权（工程 trick） | 频域 FFT + 可学习门控掩码（有理论基础） |
| **创新叙事** | "高频增强+低频平滑"（太通用，谁都能想到） | "频域选择性边界增强"（与浅层边缘蒸馏 L_edge 形成闭环） |
| **审稿人观感** | 像 trick 堆叠 | 频域+空域协同有学术新意 |
| **与主线的关系** | 独立，和 IG-Scan / 蒸馏无交互 | **与 L_edge 互补：FGEB 提取高频边界 → L_edge 用教师纠偏** |
| **文献支撑** | 无直接参考 | CFG-MambaNet（Digital Medicine, 2025）提供频域 Mamba 先例 |
| **风险** | 低（稳但无趣） | 中（需要控制高频噪声放大） |

**策略**：FGEB 为主文第三模块；AFM 仅在附录作为保守对照，证明 FGEB > AFM > None。

---

## 1. 论文标题

### 推荐标题
**RS-LightMamba: Lightweight Mamba-based Remote Sensing Instance Segmentation with Foreground-Guided Scanning, Frequency-Domain Edge Boosting, and Cross-Domain Integral Distillation from SAM**

### 备用短标题
RS-LightMamba: Efficient Remote Sensing Instance Segmentation via Dynamic Scan Ordering and Integral Distillation

---

## 2. 一句话核心

fg> 我们用一个 ~20M 参数的轻量 Mamba 编码器替代 SAM 的 ~600M ViT 编码器，通过 **前景感知的动态扫描顺序（FG-IG-Scan）**、**轻量频域边界增强（FGEB）**和 **Backbone+FPN 一体化蒸馏（Integral Distillation）**，在遥感实例分割中实现接近大模型精度、5-10× 推理加速。

---

## 3. 遥感四大痛点（论文动机锚点）

### 痛点 A：前景极度稀疏（Foreground Sparsity）
```
iSAID 前景占比 5-15%，85%+ 是无意义背景
→ SSM 隐状态被背景信息主导，前景特征指数衰减
→ 我们称之为"遥感前景状态稀释问题"（RS-FSD）
→ 解决方案：FG-IG-Scan
```

### 痛点 B：极端尺度差异（Scale Variance）
```
小型车辆 150-300 px² vs 大型飞机 4000+ px²
同一张图内面积差 100 倍
→ 小目标在 Mamba 长序列中只有几个 token
→ 解决方案：FG-IG-Scan（优先处理目标区域）+ FGEB（增强边界）
```

### 痛点 C：小目标边界模糊 + 背景高频干扰
```
遥感图像的独特矛盾：
  ① 小目标（车辆、船只）的边界只有 1-2 像素宽 → 需要增强高频
  ② 背景（海浪、农田条纹、屋顶纹理）也有大量高频 → 需要抑制无关高频
→ 不能无脑高频放大（会放大背景噪声）
→ 需要"选择性"高频增强
→ 解决方案：FGEB（频域门控 + 蒸馏纠偏）
```

### 痛点 D：SAM 域偏移（Cross-Domain Gap）
```
SAM 训练于 SA-1B（自然图像），遥感与之差异极大
→ 浅层边缘通用可迁移，深层语义需间接对齐
→ 解决方案：Integral Distillation（浅层 Edge + 深层 Channel + FPN Pyramid）
```

---

## 4. 整体框架（训练 vs 推理）

```text
━━━━━━━━━━━━━━ 训练阶段 ━━━━━━━━━━━━━━

输入遥感图像 I (800×800)
│
├──────────────────────────────────────────┐
↓                                          ↓
[SAM2 Image Encoder]                   [RS-LightMamba]
 冻结，离线预计算                        (可训练)
 存磁盘                                       │
│                                              ↓
教师多尺度特征                     ┌── LightMamba Encoder ────────┐
{T_shallow, T_deep, T_fpn}        │                               │
│                                  │ Stage 1: 标��� VSS Block       │
│                                  │    └→ ★ FGEB 插在 Stage1 后   │
│                                  │ Stage 2: 标准 VSS Block       │
│                                  │    └→ ★ FGEB 插在 Stage2 后   │
│                                  │ Stage 3: FG-IG-VSS Block      │
│                                  │ Stage 4: FG-IG-VSS Block      │
│                                  └───────────────────────────────┘
│                                              │
│                                        FPN 多尺度融合
│                                     {S_P2, S_P3, S_P4, S_P5}
│                                              │
│    ┌── Integral Distillation ──┐             │
│    │ L_edge    (浅层梯度域MSE)  │             │
│    │ L_channel (深层通道KL)     │             │
│    │ L_pyramid (FPN SmoothL1)   │             │
│    └───────────┬────────────────┘             │
│                └──── 对齐 ←────────��──────────┘
│                                              │
│                                       Mask R-CNN Head
│                                              │
│                                        实例分割结果

Loss = L_task + λ₁·L_edge + λ₂·L_channel + λ₃·L_pyramid + μ·L_fg
                                                           ↑ 前景辅助
━━━━━━━━━━━━━━ 推理阶段 ━━━━━━━━━━━━━━

I → LightMamba(FGEB + FG-IG-Scan) → FPN → Mask R-CNN → 输出
         SAM 完全丢弃！参数量 ~20M，复杂度 O(N)
```

---

## 5. 创新点①：FG-IG-Scan（前景感知的分组重要性引导扫描）—— 核心架构创新

### 5.1 解决什么问题

SSM 递推公式：`h_t = Ā·h_{t-1} + B̄·x_t`，隐状态 h_t 维度固定。遥感图像中 85%+ 的 token 是背景，导致 SSM 到达前景 token 时状态已被背景主导——**前景状态稀释问题（RS-FSD）**。

### 5.2 核心思想

**在进入 SSM 前，按区域前景概率重排序列，让包含目标的区域排在前面被 SSM 优先处理。**

与旧版 IG-Scan 的关键升级：**前景概率预测有 GT 辅助监督**（不是无监督 attention）。

### 5.3 四步流程

**Step 1：前景概率预测（有GT监督）**
```python
self.fg_head = nn.Sequential(
    nn.Conv2d(dim, dim // 4, 1),
    nn.ReLU(inplace=True),
    nn.Conv2d(dim // 4, 1, 1),
    nn.Sigmoid()
)
# 输出: fg_pred ∈ (B, 1, H, W)
# 监督: fg_gt = 所有实例 mask 合并的二值前景图
# L_fg = BCE(fg_pred, fg_gt)
# 参数增量: < 0.05M
```

**Step 2：区域级重要性 + 排序**
```python
# 将特征图分成 G×G = 4×4 = 16 个区域
fg_regions = F.adaptive_avg_pool2d(fg_pred, G)  # (B, 1, 4, 4)
region_importance = fg_regions.view(B, G*G)       # (B, 16)
sorted_idx = torch.argsort(region_importance, descending=True)

# 效果：
# 排序前: [区1(背景0.03), 区2(背景0.05), ..., 区7(飞机0.95), 区12(车辆0.87)]
# 排序后: [区7(0.95), 区12(0.87), ..., 区1(0.03), 区2(0.05)]
```

**Step 3：区域内保持局部行扫描 → SSM 处理**
```
每个区域内部仍按正常行扫描 → 保持空间局部连续性
只是区域间的顺序被重排
→ 前景token在序列前部被充分处理（SSM状态最新鲜）
→ 背景token排在后面（状态衰减无所谓）
```

**Step 4：逆排序恢复空间位置**
```python
reverse_idx = torch.argsort(sorted_idx)
# scatter 回原始空间位置 → 输出 (B, C, H, W)
```

### 5.4 正/逆交替机制

```
Stage 3 有 4 个 Block:
  Block 0 (偶数): 前景→背景（SSM状态新鲜时处理目标 → 精细边缘）
  Block 1 (奇数): 背景→前景（先积累全局上下文再看目标 → 丰富语义）
  Block 2: 前景→背景
  Block 3: 背景→前景
```

### 5.5 只在 Stage 3-4 使用

| Stage | 分辨率 | Token数 | IG-Scan | 理由 |
|---|---|---|---|---|
| 1 | H/4 × W/4 | ~40,000 | ❌ | 语义弱，fg_head 预测不准 |
| 2 | H/8 × W/8 | ~10,000 | ❌ | 开销/收益比不合适 |
| 3 | H/16 × W/16 | ~2,500 | ✅ | 语义强，16区域排序≈0开销 |
| 4 | H/32 × W/32 | ~625 | ✅ | 每token大感受野，排序价值最大 |

### 5.6 ★ 与 Samba 的精确差异（源码级分析）

Samba 的核心扫描函数 `CrossScan_saliency`：

```python name=Samba/models/encoders/vmamba.py url=https://github.com/Jia-hao999/Samba/blob/c6eec5e56922c5836c4a44ee4a685fb8541ea976/models/encoders/vmamba.py#L253-L277
# Samba 的做法：
for b in range(B):  # ← 逐样本 for 循环，无法并行！
    # 1. 用显著性图把 token 分成"显著(s)"和"非显著(ns)"两组
    s = torch.gather(x_rgb[b:b+1,:].view(1,C,H*W), 2, non_zero_indexs_expanded)
    ns, mask_ns = extract_non_zero_values(x_rgb[b:b+1,:] * (1-gt[b:b+1,:]))
    # 2. 拼接成 4 条路径
    scan_1 = torch.cat((s, ns), -1)       # 显著→非显著
    scan_2 = torch.cat((ns, s), -1)       # 非显著→显著
```

**Samba 做了什么**：
- 用二值显著性图把每个 token 独立判断"前景/背景"
- 前景 token 被逐个抽出，拼接在一起
- **破坏了空间局部性**：原本相邻的前景 token 可能被不同方向的显著性判断拆散

**精确差异表**：

| 维度 | Samba | FG-IG-Scan (Ours) |
|---|---|---|
| **排序粒度** | Token 级二值（是/否） | **区域级连续分数（0~1）** |
| **空间连续性** | ❌ 前景 token 逐个抽出，局部邻域关系丧失 | ✅ 区域内保持行扫描，局部连续性完整 |
| **计算效率** | `for b in range(B)` 逐样本循环 | `torch.argsort` + `torch.gather` 全向量化 |
| **前景监督** | 需要外部显著性 GT（另一个分支预测） | 用实例分割自带的 mask GT 合并 |
| **适用场景** | 显著性检测（单前景） | **实例分割（多前景，需区分实例）** |
| **信息保留** | 只有"是/否"两档 | 16 个区域的精细重要性排名 |

**论文用语**：
> Unlike Samba which performs token-level binary partitioning that disrupts spatial locality, our FG-IG-Scan operates at the region level with continuous importance scores, preserving local spatial coherence within each group while dynamically reordering inter-group processing priority.

### 5.7 ★ 与 MaIR 的精确差异（源码级分析）

MaIR 的核心扫描策略 `sscan()`：

```python name=MaIR/basicsr/archs/shift_scanf_util.py url=https://github.com/XLearning-SCU/2025-CVPR-MaIR/blob/07d58af10175f52bba0bb6f82edba1ea5d891e12/basicsr/archs/shift_scanf_util.py#L68-L80
# MaIR 的做法：
def sscan(inp, scan_len, shift_len=0):
    B, C, H, W = inp.shape
    # 按固定宽度 scan_len 把图像切成竖条
    # 每个竖条内做 S 形扫描（奇数列翻转）
    if shift_len == 0:
        for i in range(1, (W // scan_len)+1, 2):
            inp[:, :, :, i*scan_len:(i+1)*scan_len] = \
                inp[:, :, :, i*scan_len:(i+1)*scan_len].flip(dims=[-2])
```

MaIR Block 间的 shift 交替逻辑：

```python name=MaIR/basicsr/archs/mair_arch.py url=https://github.com/XLearning-SCU/2025-CVPR-MaIR/blob/07d58af10175f52bba0bb6f82edba1ea5d891e12/basicsr/archs/mair_arch.py#L376-L390
# 偶数Block用标准NSS路径，奇数Block用shift半个stripe的NSS路径
if self.shift_size > 0:
    x = input*self.skip_scale + self.drop_path(
        self.self_attention(x, (mair_ids[2], mair_ids[3])))  # shift版路径
else:
    x = input*self.skip_scale + self.drop_path(
        self.self_attention(x, (mair_ids[0], mair_ids[1])))  # 标准路径
```

**MaIR 做了什么**：
- **Nested S-shaped Scanning (NSS)**：固定宽度竖条 + S形扫描，增强空间连续性
- **Block 间交替 shift**：偶数Block标准路径，奇数Block偏移半个stripe
- **核心目标**：让空间相邻的像素在序列中也相邻（图像复原的核心需求）

**精确差异**：

| 维度 | MaIR | FG-IG-Scan (Ours) |
|---|---|---|
| **数据相关性** | ❌ 完全固定（scan_len是超参数，所有图像一样） | ✅ 数据相关（每张图的排序不同） |
| **优化维度** | 路径几何形状（局部性+连续性） | **处理优先级（前景优先）** |
| **解决的问题** | 空间不连续 | **前景状态稀释** |
| **适用领域** | 图像超分/去噪（全图同等重要） | **遥感实例分割（前景/背景极不均衡）** |

**论文用语**：
> MaIR focuses on scan path geometry (locality and continuity), which is data-independent. In contrast, our FG-IG-Scan is data-dependent: it dynamically reorders region processing priority based on foreground probability, addressing the unique foreground-sparsity challenge in remote sensing.

### 5.8 三代扫描策略演进表（Related Work 直接用）

| 代际 | 代表 | 改什么 | 数据相关？ | 保持空间连续？ | 解决的问题 |
|---|---|---|---|---|---|
| 1st | VMamba (2024) | 固定四方向行列扫描 | ❌ | 部分 | 2D→1D 展开 |
| 2nd-A | MaIR (CVPR'25) | S形条带+shift交替 | ❌ | ✅ | 局部性+连续性 |
| 2nd-B | Samba (CVPR'25) | 二值前景/背景token重组 | ✅ | ❌ | 显著性引导 |
| **3rd** | **FG-IG-Scan (Ours)** | **区域级连续分数排序** | **✅** | **✅** | **遥感前景稀疏** |

### 5.9 完整代码

```python
class ForegroundAwareGroupScan(nn.Module):
    """
    FG-IG-Scan: 前景感知的分组重要性引导扫描
    绑定遥感痛点: 前景稀疏(A) + 尺度差异(B)
    """
    def __init__(self, dim, num_groups=4, reverse=False):
        super().__init__()
        self.num_groups = num_groups
        self.reverse = reverse  # 偶数Block=False, 奇数Block=True
        
        # 轻量前景预测头（有GT辅助监督）
        self.fg_head = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        x: (B, C, H, W)
        Returns:
            x_reordered: (B, C, H, W) 按前景重要性重排后的特征
            fg_pred: (B, 1, H, W) 前景概率图（用于辅助Loss）
            reverse_idx: (B, G²) 逆排序索引（用于SSM后恢复）
        """
        B, C, H, W = x.shape
        g = self.num_groups
        pH, pW = H // g, W // g
        
        # 1. 预测前景概率
        fg_pred = self.fg_head(x)  # (B, 1, H, W)
        
        # 2. 区域级重要性
        fg_regions = F.adaptive_avg_pool2d(fg_pred, g)  # (B, 1, g, g)
        region_importance = fg_regions.view(B, g * g)     # (B, G²)
        
        # 3. 排序（正序或逆序交替）
        sorted_idx = torch.argsort(
            region_importance, dim=-1, 
            descending=(not self.reverse)  # 偶数Block: 重要→不重要; 奇数Block: 反向
        )
        reverse_idx = torch.argsort(sorted_idx, dim=-1)
        
        # 4. 将特征图切成 G×G 个区域块
        regions = x.reshape(B, C, g, pH, g, pW)
        regions = regions.permute(0, 2, 4, 1, 3, 5)      # (B, g, g, C, pH, pW)
        regions = regions.reshape(B, g * g, C, pH, pW)    # (B, G², C, pH, pW)
        
        # 5. 按前景重要性重排区域
        idx_expand = sorted_idx[:, :, None, None, None].expand(-1, -1, C, pH, pW)
        regions_sorted = torch.gather(regions, 1, idx_expand)
        
        # 6. 恢复为 2D 送入 SSM（区域内保持行扫描）
        x_reordered = regions_sorted.reshape(B, g, g, C, pH, pW)
        x_reordered = x_reordered.permute(0, 3, 1, 4, 2, 5)  # (B,C,g,pH,g,pW)
        x_reordered = x_reordered.reshape(B, C, H, W)
        
        return x_reordered, fg_pred, reverse_idx
    
    def inverse_reorder(self, x, reverse_idx):
        """SSM处理后，恢复原始空间位置"""
        B, C, H, W = x.shape
        g = self.num_groups
        pH, pW = H // g, W // g
        
        regions = x.reshape(B, C, g, pH, g, pW)
        regions = regions.permute(0, 2, 4, 1, 3, 5).reshape(B, g*g, C, pH, pW)
        
        rev_expand = reverse_idx[:, :, None, None, None].expand(-1, -1, C, pH, pW)
        regions_restored = torch.gather(regions, 1, rev_expand)
        
        regions_restored = regions_restored.reshape(B, g, g, C, pH, pW)
        regions_restored = regions_restored.permute(0, 3, 1, 4, 2, 5)
        return regions_restored.reshape(B, C, H, W)
    
    def compute_fg_loss(self, fg_pred, fg_gt):
        """前景辅助Loss"""
        fg_gt_resized = F.interpolate(fg_gt, size=fg_pred.shape[2:], mode='nearest')
        return F.binary_cross_entropy(fg_pred, fg_gt_resized)


class FG_IG_VSSBlock(nn.Module):
    """替换 Stage 3-4 的标准 VSSBlock"""
    def __init__(self, dim, num_groups=4, reverse=False, **vss_kwargs):
        super().__init__()
        self.ig_scan = ForegroundAwareGroupScan(dim, num_groups, reverse)
        self.norm = nn.LayerNorm(dim)
        self.ssm = SS2D(d_model=dim, **vss_kwargs)  # 复用VMamba的SS2D
        self.mlp = Mlp(in_features=dim, hidden_features=dim*4)
        self.drop_path = DropPath(vss_kwargs.get('drop_path', 0.))
    
    def forward(self, x, fg_gt=None):
        """
        x: (B, C, H, W)
        fg_gt: (B, 1, H_orig, W_orig) 前景GT（训练时）
        """
        residual = x
        
        # 1. FG-IG-Scan 重排
        x_reordered, fg_pred, reverse_idx = self.ig_scan(x)
        
        # 2. SSM 处理（在重排后的序列上）
        x_ssm = self.ssm(self.norm(x_reordered.permute(0,2,3,1)).permute(0,3,1,2))
        
        # 3. 逆排序恢复空间
        x_restored = self.ig_scan.inverse_reorder(x_ssm, reverse_idx)
        
        # 4. 残差 + MLP
        x = residual + self.drop_path(x_restored)
        x = x + self.drop_path(self.mlp(self.norm(x.permute(0,2,3,1)).permute(0,3,1,2)))
        
        # 5. 计算前景辅助Loss
        fg_loss = None
        if fg_gt is not None and self.training:
            fg_loss = self.ig_scan.compute_fg_loss(fg_pred, fg_gt)
        
        return x, fg_pred, fg_loss
```

### 5.10 开销分析

| 指标 | 数值 |
|---|---|
| 新增参数（fg_head） | ~0.01M per block |
| 排序开销 | argsort 16 个数 ≈ 0 |
| gather/scatter | O(H×W)，可忽略 |
| 推理延迟增加 | < 1% |

---

## 6. 创新点②：Integral Distillation（Backbone + FPN 一体化蒸馏）—— 核心训练创新

### 6.1 为什么需要 Backbone + FPN 都蒸馏

受 **Fast-iTPN**（TPAMI 2024）启发。iTPN 的核心主张：backbone 和金字塔应该 **整体（integrally）** 优化，而非分别训练。

```python name=iTPN/itpn_clip/modeling_pretrain.py url=https://github.com/sunsmarterjie/iTPN/blob/638eff48799760e0afda68812427cf406a0e13fd/itpn_clip/modeling_pretrain.py#L23-L33
class iTPNClipDistill(nn.Module):
    # iTPN 在预训练阶段就把 FPN 纳入蒸馏：
    # backbone features → FPN → 多层级输出 → 与教师CLIP对齐
    def __init__(self, ..., fpn_dim=256, fpn_depth=2, teacher_dim=512, ...):
```

**只蒸馏 backbone 的问题**：
- Backbone 特征改善了，但 FPN 从零学起
- FPN 的 P2/P3 层直接影响小目标（遥感核心场景）
- 信息在 backbone→FPN 的传递中产生"语义断层"

### 6.2 三层蒸馏体系

#### (a) 浅层边缘蒸馏 L_edge

```
目标：迁移 SAM 的边界检测能力（跨域安全——边缘是通用的低级特征）

L_edge = MSE( Sobel(ϕ_s(F_S^shallow)), Sobel(F_T^shallow) )

ϕ_s = 1×1 Conv adapter（学生通道→教师通道）
Sobel = 固定 Sobel 算子，提取梯度幅值

不在原始特征域做 MSE（SAM 的自然图像特征值分布和遥感不同）
而在梯度域做 MSE（"哪里有边缘"这个信息跨域通用）

★ 与 FGEB 的协同：
  FGEB 在频域增强了学生特征的高频分量
  L_edge 用教师的边缘知识纠正 FGEB 可能引入的高频噪声
  → "FGEB 提取 + L_edge 纠偏"形成闭环
```

#### (b) 深层通道蒸馏 L_channel

```
目标：迁移"哪些语义通道更重要"的相对关系（跨域鲁棒）

z_S = GAP(ϕ_d(F_S^deep))    → (B, C)
z_T = GAP(F_T^deep)          → (B, C)
p_S = softmax(z_S / τ)       → 通道重要性分布
p_T = softmax(z_T / τ)
L_channel = KL(p_S ‖ p_T) × τ²

τ = 4.0（温度系数，软化分布，增强跨域鲁棒性）

不要求学生的通道值和教师一样大
只要求通道间的相对重要性分布一致
```

#### (c) FPN 金字塔蒸馏 L_pyramid（受 iTPN 启发，新增）

```
目标：直接监督 FPN 输出层 P2-P5

教师 FPN 特征来源：
  SAM2 的 Hiera backbone 本身就是多尺度的
  取中间特征做线性投影到 P2-P5 尺寸

L_pyramid = Σ_{l=2}^{5} w_l · SmoothL1(ψ_l(S_Pl), T_Pl)

ψ_l = 层级适配器（1×1 Conv）
w_l = 层级权重（建议 w2=w3=1.0, w4=w5=0.5，小目标更重要）
```

### 6.3 总 Loss

```
L_total = L_task + λ₁·L_edge + λ₂·L_channel + λ₃·L_pyramid + μ·L_fg

L_task  = L_cls + L_reg + L_mask （Mask R-CNN 标准任务Loss）
L_fg    = BCE（FG-IG-Scan 的前景预测辅助Loss）

超参数（建议初值）：
  λ₁ = 0.5    浅层边缘蒸馏
  λ₂ = 0.5    深层通道蒸馏
  λ₃ = 1.0    FPN金字塔蒸馏
  τ  = 4.0    KL温度
  μ  = 0.3    前景辅助
```

### 6.4 训练 vs 推理

| | 训练 | 推理 |
|---|---|---|
| SAM2 教师 | 离线预提取特征存磁盘 | **完全丢弃** |
| 蒸馏 Loss | 计算并回传梯度 | **不存在** |
| 额外开销 | 读 .npz 文件 | **零** |

### 6.5 与 MobileSAM 蒸馏的差异

| | MobileSAM | Ours |
|---|---|---|
| 学生 | TinyViT (Transformer) | LightMamba (SSM, O(N)) |
| 蒸馏层次 | 仅最后一层 | 浅层Edge + 深层Channel + FPN |
| Loss 类型 | 直接 MSE | 梯度域MSE + 通道KL + SmoothL1 |
| FPN 监督 | 无 | ✅ 有 |
| 域适配 | 无 | 有（跨域安全的间接特征对齐）|

### 6.6 蒸馏代码

```python
class IntegralDistillation(nn.Module):
    """
    Backbone + FPN 一体化蒸馏
    受 iTPN(TPAMI'24) + CFG-MambaNet 启发
    """
    def __init__(self, student_dims, teacher_dims, fpn_channels=256, tau=4.0):
        super().__init__()
        self.tau = tau
        
        # Backbone adapters
        self.shallow_adapter = nn.Conv2d(student_dims[0], teacher_dims[0], 1)
        self.deep_adapter = nn.Conv2d(student_dims[-1], teacher_dims[-1], 1)
        
        # FPN adapters (P2-P5)
        self.fpn_adapters = nn.ModuleList([
            nn.Conv2d(fpn_channels, fpn_channels, 1) for _ in range(4)
        ])
        
        # 固定 Sobel 核
        sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1,1,3,3))
        self.register_buffer('sobel_y', sobel_y.view(1,1,3,3))
    
    def _sobel_gradient(self, feat):
        feat_mean = feat.mean(dim=1, keepdim=True)
        gx = F.conv2d(feat_mean, self.sobel_x, padding=1)
        gy = F.conv2d(feat_mean, self.sobel_y, padding=1)
        return torch.sqrt(gx**2 + gy**2 + 1e-6)
    
    def edge_loss(self, f_s, f_t):
        f_s = self.shallow_adapter(f_s)
        return F.mse_loss(self._sobel_gradient(f_s), self._sobel_gradient(f_t))
    
    def channel_loss(self, f_s, f_t):
        f_s = self.deep_adapter(f_s)
        z_s = F.adaptive_avg_pool2d(f_s, 1).flatten(1)
        z_t = F.adaptive_avg_pool2d(f_t, 1).flatten(1)
        p_s = F.log_softmax(z_s / self.tau, dim=-1)
        p_t = F.softmax(z_t / self.tau, dim=-1)
        return F.kl_div(p_s, p_t, reduction='batchmean') * (self.tau ** 2)
    
    def pyramid_loss(self, student_fpn, teacher_fpn):
        """FPN 层级蒸馏"""
        weights = [1.0, 1.0, 0.5, 0.5]  # P2,P3权重高（小目标）
        loss = 0
        for i, (s, t, w, adapter) in enumerate(
            zip(student_fpn, teacher_fpn, weights, self.fpn_adapters)):
            s_aligned = adapter(s)
            # 尺寸对齐
            if s_aligned.shape != t.shape:
                t = F.interpolate(t, size=s_aligned.shape[2:], mode='bilinear', align_corners=False)
            loss += w * F.smooth_l1_loss(s_aligned, t)
        return loss
    
    def forward(self, student_backbone, teacher_backbone, student_fpn=None, teacher_fpn=None):
        l_edge = self.edge_loss(student_backbone['shallow'], teacher_backbone['shallow'])
        l_chan = self.channel_loss(student_backbone['deep'], teacher_backbone['deep'])
        l_pyr = 0
        if student_fpn is not None and teacher_fpn is not None:
            l_pyr = self.pyramid_loss(student_fpn, teacher_fpn)
        return l_edge, l_chan, l_pyr
```

---

## 7. 创新点③：FGEB（Frequency-Guided Edge Booster）—— 轻量频域边界增强

**这是你读 CFG-MambaNet 后得到的灵感，也是本方案与旧版最大的区别。AFM 是保守备选，FGEB 才是正式模块。**

### 7.1 角色定位

| 属性 | 说明 |
|---|---|
| **定位** | 第三模块，但严格轻量化 |
| **目标** | 增强小目标边界的高频分量，同时抑制背景高频噪声 |
| **插入位置** | Stage 1/2 输出后（浅层，高分辨率，边界信息最丰富） |
| **核心创新** | **可学习频域门控掩码** + **L_edge 蒸馏纠偏协同** |
| **与 AFM 的区别** | AFM 是空间域三分支加权（工程 trick），FGEB 是频域选择性增强（有理论基础） |

### 7.2 遥感专属动机

```
遥感图像的独特频域矛盾：
  ① 小目标边界 = 有用高频（需要增强）
     车辆轮廓（150-300px²）、船只边缘（50-500px²）
     这些高频信号对实例分割的 mask 质量至关重要
  
  ② 背景纹理 = 有害高频（需要抑制）
     海浪波纹、农田沟渠条纹、屋顶瓦片纹理
     这些高频噪声会干扰边缘检测

  传统做法（如 AFM 的 Laplacian）：
     → 无差别放大所有高频 → 背景噪声也被放大
  
  FGEB 的做法：
     → 在频域用可学习门控"选择性地"增强有用高频
     → 同时抑制无关高频
     → 再由 L_edge 蒸馏用教师知识纠偏
```

### 7.3 技术方案（详细）

```text
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  FGEB: Frequency-Guided Edge Booster
  插入位置：Stage 1/2 输出后
━━━━━━━━━━━━━━━━━━━━��━━━━━━━━━━━━━━━━

输入特征 F ∈ R^{B×C×H×W}（来自 Stage 1 或 Stage 2 输出）
    │
    ↓
┌─────────────────────────────────────────────────────┐
│ Step 1: FFT 变换到频域                                │
│                                                       │
│   F_freq = torch.fft.rfft2(F, norm='ortho')          │
│   F_freq ∈ C^{B×C×H×(W/2+1)}  （复数频谱）          │
│   F_amp = |F_freq|              （幅度谱）            │
│   F_phase = angle(F_freq)       （相位谱）            │
│                                                       │
│ 为什么用 FFT？                                        │
│   空间域的"边缘"对应频域的"高频分量"                   │
│   在频域操作可以精确控制不同频段的增强/抑制              │
└─────────────────────────────────────────────────────┘
    │
    ↓
┌─────────────────────────────────────────────────────┐
│ Step 2: 可学习频域门控掩码 M_gate                     │
│                                                       │
│ 方案A（推荐）：通道级可学习带通滤波器                   │
│                                                       │
│   # 生成频率坐标网格 [0,1]                            │
│   freq_h = torch.linspace(0, 1, H)                   │
│   freq_w = torch.linspace(0, 1, W//2+1)              │
│   freq_grid = sqrt(freq_h² + freq_w²)  → 频率距离图   │
│                                                       │
│   # 可学习的带通参数（每通道独立）                      │
│   self.center = nn.Parameter(0.3*ones(C))  → 带通中心  │
│   self.width  = nn.Parameter(0.2*ones(C))  → 带通宽度  │
│                                                       │
│   # 高斯带通掩码                                      │
│   M_bandpass = exp(-(freq_grid - center)²/(2·width²))  │
│   → 自动学习哪个频段需要增强                            │
│   → 边界频段(0.2-0.5)的权重会被学大                     │
│   → 超高频噪声(0.8-1.0)的权重会被学小                   │
│                                                       │
│ 方案B（备选）：轻量卷积门控                             │
│   # 在频域幅度谱上做 1×1 Conv → Sigmoid                │
│   M_gate = sigmoid(Conv1x1(F_amp))                    │
│   → 更灵活但参数更多                                   │
│                                                       │
│ ★ 门控的关键作用：                                     │
│   高频中"边界对应的频段" → 权重大 → 增强                │
│   高频中"噪声对应的频段" → 权重小 → 抑制                │
│   低频（背景平坦区域）  → 权重≈0 → 不变                 │
└─────────────────────────────────────────────────────┘
    │
    ↓
┌─────────────────────────────────────────────────────┐
│ Step 3: 频域滤波 + IFFT 回空间域                       │
│                                                       │
│   F_filtered = F_freq ⊙ M_gate                        │
│   F_high = torch.fft.irfft2(F_filtered, norm='ortho') │
│   → F_high 包含了"选择性增强"的高频分量                 │
└─────────────────────────────────────────────────────┘
    │
    ↓
┌─────────────────────────────────────────────────────┐
│ Step 4: 残差融合（α可学习，小初始化）                   │
│                                                       │
│   F_out = F + α · F_high                              │
│   α = nn.Parameter(tensor(0.1))                       │
│                                                       │
│ 为什么小初始化 α=0.1？                                │
│   → 训练初期 FGEB 几乎不改变特征（稳定训练）            │
│   → 随着训练进行，α 自动调整到合适的增强强度             │
│   → 如果 FGEB 无用，α 会趋近于 0（自动关断）            │
└─────────────────────────────────────────────────────┘
    │
    ↓
输出 F_out ∈ R^{B×C×H×W}（边界增强后的特征）
```

### 7.4 ★ FGEB 与 L_edge 的协同闭环

```
这是 FGEB 作为第三模块的核心叙事价值：

┌─────────────┐     ┌───────────��──────┐
│   FGEB      │ ──→ │ 学生浅层特征      │
│ 频域边界增强 │     │ (高频分量被增强)   │
└─────────────┘     └────────┬─────────┘
                             │
                    ┌────────↓─────────┐
                    │   L_edge 蒸馏     │
                    │ 教师的边缘知识     │
                    │ 纠正 FGEB 可能    │
                    │ 引入的噪声        │
                    └──────────────────┘

分工：
  FGEB → "提取"：在频域选择性增强目标边界的高频信号
  L_edge → "纠偏"：用 SAM 教师的边缘检测能力约束增强质量

如果没有 L_edge：FGEB 可能放大背景噪声（无约束）
如果没有 FGEB：L_edge 只能在原始特征上约束（缺少增强手段）
两者协同 → 学生既有增强的边界特征，又有教师级别的边缘精度
```

### 7.5 完整代码

```python
class FrequencyGuidedEdgeBooster(nn.Module):
    """
    FGEB: 轻量频域边界增强模块
    绑定遥感痛点: 小目标边界模糊(B) + 背景高频干扰(C)
    受 CFG-MambaNet (Digital Medicine 2025) 启发
    """
    def __init__(self, dim, mode='bandpass'):
        """
        Args:
            dim: 输入特征通道数
            mode: 'bandpass' (推荐，可学习带通) 或 'conv_gate' (卷积门控)
        """
        super().__init__()
        self.dim = dim
        self.mode = mode
        
        if mode == 'bandpass':
            # 可学习带通滤波器参数（每通道独立）
            self.center = nn.Parameter(torch.full((dim,), 0.3))  # 带通中心频率
            self.width = nn.Parameter(torch.full((dim,), 0.2))   # 带通宽度
        elif mode == 'conv_gate':
            # 在频域幅度谱上做轻量门控
            self.gate_net = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim // 4, dim, 1),
                nn.Sigmoid()
            )
        
        # 残差融合系数（小初始化 → 训练初期几乎不改变特征）
        self.alpha = nn.Parameter(torch.tensor(0.1))
    
    def _build_freq_grid(self, H, W, device):
        """构建归一化频率距离网格 [0, 1]"""
        freq_h = torch.linspace(0, 1, H, device=device)
        freq_w = torch.linspace(0, 1, W, device=device)
        grid_h, grid_w = torch.meshgrid(freq_h, freq_w, indexing='ij')
        freq_dist = torch.sqrt(grid_h**2 + grid_w**2).clamp(max=1.0)
        return freq_dist  # (H, W)
    
    def forward(self, x):
        """
        x: (B, C, H, W) 来自 Stage 1/2 的特征
        Returns: (B, C, H, W) 边界增强后的特征
        """
        B, C, H, W = x.shape
        
        # 1. FFT 到频域
        x_freq = torch.fft.rfft2(x, norm='ortho')  # (B, C, H, W//2+1), complex
        x_amp = x_freq.abs()                         # 幅度谱
        
        W_freq = x_freq.shape[-1]  # W//2+1
        
        if self.mode == 'bandpass':
            # 2a. 可学习高斯带通掩码
            freq_grid = self._build_freq_grid(H, W_freq, x.device)  # (H, W_freq)
            
            # center 和 width 扩展为 (C, 1, 1) 以支持逐通道操作
            center = self.center.view(C, 1, 1).clamp(0.05, 0.95)
            width = self.width.view(C, 1, 1).clamp(0.05, 0.5)
            
            # 高斯带通: 在 center 附近的频率被增强
            freq_grid_expand = freq_grid.unsqueeze(0)  # (1, H, W_freq)
            mask = torch.exp(-((freq_grid_expand - center)**2) / (2 * width**2))
            # mask: (C, H, W_freq), 值域 [0, 1]
            mask = mask.unsqueeze(0)  # (1, C, H, W_freq)
            
        elif self.mode == 'conv_gate':
            # 2b. 在幅度谱上做卷积门控
            mask = self.gate_net(x_amp)  # (B, C, H, W_freq)
        
        # 3. 频域滤波
        x_filtered = x_freq * mask
        
        # 4. IFFT 回空间域
        x_high = torch.fft.irfft2(x_filtered, s=(H, W), norm='ortho')
        
        # 5. 残差融合
        out = x + self.alpha * x_high
        
        return out


class FrequencyGuidedEdgeBooster_DoG(nn.Module):
    """
    FGEB 的 Difference-of-Gaussian 变体（更简单，更稳定）
    不需要 FFT，纯空间域实现，但本质是频域带通滤波的近似
    """
    def __init__(self, dim):
        super().__init__()
        # 两个不同 σ 的高斯核 → DoG ≈ 带通滤波
        self.sigma1 = nn.Parameter(torch.tensor(1.0))  # 小σ（保留高频）
        self.sigma2 = nn.Parameter(torch.tensor(3.0))  # 大σ（模糊高频）
        
        # 门控：控制增强强度
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        
        self.alpha = nn.Parameter(torch.tensor(0.1))
    
    def _gaussian_blur(self, x, sigma):
        """可微的高斯模糊"""
        k_size = max(3, int(2 * sigma.item()) * 2 + 1)
        k_size = min(k_size, 7)  # 限制核大小
        padding = k_size // 2
        
        # 1D 高斯核
        coords = torch.arange(k_size, dtype=torch.float32, device=x.device) - k_size // 2
        kernel_1d = torch.exp(-coords**2 / (2 * sigma**2 + 1e-6))
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        # 可分离卷积
        C = x.shape[1]
        kernel_h = kernel_1d.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
        kernel_v = kernel_1d.view(1, 1, 1, -1).repeat(C, 1, 1, 1)
        
        x = F.conv2d(x, kernel_h, padding=(padding, 0), groups=C)
        x = F.conv2d(x, kernel_v, padding=(0, padding), groups=C)
        return x
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # DoG = Gaussian(σ1) - Gaussian(σ2) ≈ 带通滤波
        g1 = self._gaussian_blur(x, self.sigma1)
        g2 = self._gaussian_blur(x, self.sigma2)
        dog = g1 - g2  # 带通分量
        
        # 门控
        gate_weight = self.gate(x).view(B, C, 1, 1)
        dog_gated = dog * gate_weight
        
        out = x + self.alpha * dog_gated
        return out
```

### 7.6 两种 FGEB 实现的对比

| | FFT 版（推荐） | DoG 版（备选） |
|---|---|---|
| **理论基础** | 精确频域操作 | DoG ≈ 带通滤波的空间域近似 |
| **参数** | 2×C (center+width) + 1 (alpha) ≈ 0.001M | ~0.01M |
| **FFT 开销** | Stage1/2 各调用 1 次 rfft2+irfft2 | 无 FFT，两次高斯卷积 |
| **频率控制精度** | 高（精确指定频段） | 中（DoG 只有一个带通） |
| **稳定性** | 中（需要 clamp center/width） | 高 |
| **消融实验** | 主表格 | 表3 备选 |

**建议**：主文报告 FFT 版结果。如果 FFT 版不稳定（训练 NaN 或无增益），切换到 DoG 版。

### 7.7 FGEB 的风险控制

| 风险 | 应对 |
|---|---|
| 高频噪声误增强 | ① 门控掩码限制增强频段 ② L_edge 蒸馏纠偏 ③ α小初始化 |
| FFT 导致训练 NaN | ① `norm='ortho'` 正则化 ② clamp center/width ③ 切 DoG 版 |
| 推理速度下降 | ① 仅 Stage1/2（高分辨率但只调用 2 次）② FFT 在 GPU 上极快 |
| 增益太小 | ① 先测 FFT 版 vs DoG 版 vs None ② 如均无增益则降为附录 |

### 7.8 FGEB 在网络中的插入位置

```python
class LightMambaBackbone(nn.Module):
    def __init__(self, dims=[48,96,192,384], depths=[2,2,4,1]):
        ...
        # Stage 1
        self.stage1 = nn.Sequential(*[VSSBlock(dim=dims[0]) for _ in range(depths[0])])
        self.fgeb1 = FrequencyGuidedEdgeBooster(dim=dims[0])  # ★ Stage1 后插入
        self.downsample1 = PatchMerging2D(dims[0], dims[1])
        
        # Stage 2
        self.stage2 = nn.Sequential(*[VSSBlock(dim=dims[1]) for _ in range(depths[1])])
        self.fgeb2 = FrequencyGuidedEdgeBooster(dim=dims[1])  # ★ Stage2 后插入
        self.downsample2 = PatchMerging2D(dims[1], dims[2])
        
        # Stage 3 (FG-IG-Scan)
        self.stage3 = nn.ModuleList([
            FG_IG_VSSBlock(dim=dims[2], reverse=(i%2==1))
            for i in range(depths[2])
        ])
        # Stage 3 不插入 FGEB（深层语义 > 边界细节）
        ...
    
    def forward(self, x, fg_gt=None):
        # Stage 1
        x = self.stage1(x)
        x = self.fgeb1(x)          # ★ 频域边界增强
        feats = [x]                 # 浅层特征（给 L_edge 用）
        x = self.downsample1(x)
        
        # Stage 2
        x = self.stage2(x)
        x = self.fgeb2(x)          # ★ 频域边界增强
        feats.append(x)
        x = self.downsample2(x)
        
        # Stage 3-4 (FG-IG-Scan)
        fg_losses = []
        for blk in self.stage3:
            x, fg_pred, fg_loss = blk(x, fg_gt)
            if fg_loss is not None:
                fg_losses.append(fg_loss)
        feats.append(x)
        ...
        return feats, fg_losses
```

### 7.9 与 CFG-MambaNet 的差异（审稿人可能追问）

| | CFG-MambaNet | FGEB (Ours) |
|---|---|---|
| **任务** | 医学图像分割 | **遥感实例分割** |
| **频域模块规模** | 较重（多层 FFT + 上下文模块） | **轻量（仅 2 个参数向量 + 1 个 α）** |
| **频域操作位置** | 嵌入 Mamba Block 内部 | **Stage 输��后（不改 VSSBlock 内部）** |
| **是否有蒸馏纠偏** | 否 | **是（L_edge 与 FGEB 协同闭环）** |
| **门控类型** | 上下文引导门控 | **可学习带通参数（更轻更可解释）** |

**论文用语**：
> Inspired by CFG-MambaNet's frequency-domain design for medical image segmentation, we propose a lightweight variant FGEB tailored for remote sensing. Unlike CFG-MambaNet's heavyweight contextual frequency modules embedded inside Mamba blocks, FGEB operates as a post-stage plugin with learnable bandpass parameters and minimal overhead. Crucially, FGEB forms a closed loop with our edge-aware distillation: FGEB enhances boundary-sensitive frequency components, while L_edge leverages teacher knowledge to rectify potential noise amplification.

---

## 8. AFM 的定位（附录对照，不是主文贡献）

```
AFM（旧版第三模块）保留为消融实验中的保守对照：

消融表中的一行：
  | None  | 无预处理模块         | xx.x mAP |
  | AFM   | 空间域四方向Sobel融合 | xx.x mAP |
  | FGEB  | 频域可学习带通增强    | xx.x mAP |  ← 主表格

如果 FGEB > AFM > None → 完美
如果 FGEB ≈ AFM > None → 仍可叙述"频域更轻量且可解释"
如果 AFM > FGEB         → 切换回 AFM 作为主模块（保底）
```

---

## 9. 论文三大贡献的标准表述

> **(1)** We propose **FG-IG-Scan**, a foreground-guided importance group scanning mechanism for visual state space models. Unlike existing fixed-order scans (VMamba), geometry-optimized paths (MaIR), or token-level binary partitioning (Samba), FG-IG-Scan performs **data-dependent region-level reordering** with GT-supervised foreground probability, preserving local spatial continuity while prioritizing target-rich regions. This directly addresses the foreground state dilution problem unique to remote sensing.

> **(2)** We design an **Integral Distillation** framework that jointly distills **shallow edge representations** (gradient-domain MSE), **deep channel semantics** (temperature-softened KL), and **FPN pyramid features** (SmoothL1) from SAM2 to our lightweight Mamba encoder. Inspired by iTPN's integral pre-training philosophy, this backbone-to-pyramid pipeline ensures comprehensive knowledge transfer while being robust to the natural-to-remote-sensing domain gap.

> **(3)** We introduce **FGEB (Frequency-Guided Edge Booster)**, a lightweight frequency-domain module that selectively enhances boundary-sensitive high-frequency components via learnable bandpass gating, while forming a synergistic closed loop with edge-aware distillation. Comprehensive experiments on iSAID and NWPU VHR-10 demonstrate that RS-LightMamba achieves competitive accuracy with ~3% of SAM's parameters and ~5-10× inference speedup.

---

## 10. 消融实验设计（完整版）

### 表1：各模块增量消融
| # | 配置 | segm mAP | AP_s | Params | FPS | 验证什么 |
|---|---|---|---|---|---|---|
| A | LightMamba Baseline | | | ~18M | | 基准 |
| B | A + FGEB | | | ~18M | | FGEB 单独增益 |
| C | A + FG-IG-Scan | | | ~18M | | FG-IG-Scan 单独增益 |
| D | A + Backbone 蒸馏 (Edge+Channel) | | | ~18M | | Backbone 蒸馏效果 |
| E | A + Integral 蒸馏 (+FPN) | | | ~18M | | FPN 蒸馏额外增益 |
| F | A + FGEB + FG-IG-Scan | | | ~18M | | 架构两模块叠加 |
| G | **F + Integral 蒸馏（完整模型）** | | | **~20M** | | **完整模型** |

### 表2：★ 防撞车消融（IG-Scan vs Samba vs MaIR vs 随机）
| # | 扫描策略 | segm mAP | AP_s | 说明 |
|---|---|---|---|---|
| H1 | 标准四方向 (VMamba) | | | Baseline |
| H2 | **Samba 式二值分割** | | | Token级前景/背景重排 |
| H3 | 随机区域排序 | | | 排除正则化效应 |
| H4 | IG-Scan（无前景GT监督） | | | 无监督版 |
| H5 | **FG-IG-Scan（有前景GT监督）** | | | 完整方案 |

> H2 vs H5 → 区域级排序 > token级二值分割
> H3 vs H5 → 语义排序 > 随机排序
> H4 vs H5 → GT 监督的必要性

### 表3：FGEB 消融
| # | 边界增强方式 | segm mAP | AP_s |
|---|---|---|---|
| I1 | None | | |
| I2 | AFM（空间域，旧版） | | |
| I3 | FGEB-DoG（空间域近似） | | |
| I4 | FGEB-FFT 无门控（全频增强） | | |
| I5 | **FGEB-FFT + 可学习带通门控** | | |

> I4 vs I5 → 门控的必要性（无门控=放大所有高频=放大噪声）
> I2 vs I5 → FGEB > AFM（频域 > 空间域）
> I3 vs I5 → FFT版 vs DoG版

### 表4：蒸馏策略消融
| # | 蒸馏方式 | segm mAP |
|---|---|---|
| J1 | 无蒸馏 | |
| J2 | 单层MSE (MobileSAM式) | |
| J3 | 多层MSE | |
| J4 | Edge + Channel (Backbone only) | |
| J5 | **Edge + Channel + Pyramid (Integral)** | |

### 表5：与 SOTA 对比
| 方法 | Backbone | segm mAP | AP_s | Params | FPS |
|---|---|---|---|---|---|
| Mask R-CNN | R-50 | | | 44M | |
| Mask R-CNN | R-101 | | | 63M | |
| Cascade Mask R-CNN | R-50 | | | 77M | |
| Mask R-CNN | Swin-T | | | ~48M | |
| RSPrompter | SAM-ViT-B | | | 100M+ | |
| MobileSAM + MaskRCNN | TinyViT | | | ~10M | |
| **RS-LightMamba (Ours)** | **LightMamba** | | | **~20M** | |

---

## 11. LightMamba 编码器配置

| 参数 | VMamba-Tiny（参考） | LightMamba（你的） |
|---|---|---|
| 通道数 | [96, 192, 384, 768] | [48, 96, 192, 384] |
| Block 数 | [2, 2, 9, 2] | [2, 2, 4, 1] |
| 卷积类型 | 标准 3×3 Conv | 深度可分离 Conv |
| Stage 1-2 后 | 无 | **+ FGEB** |
| Stage 3-4 | 标准 VSS Block | **FG-IG-VSS Block** |
| 参数量 | ~30M | ~15-20M |
| FPN in_channels | [96,192,384,768] | [48,96,192,384] |

---

## 12. 风险清单与应对

| 风险 | 概率 | 应对 |
|---|---|---|
| FG-IG-Scan 收益不稳定 | 中 | ① Identity test ② 仅 Stage4 先验 ③ 检查fg_head可视化 |
| FGEB FFT 训练 NaN | 中 | ① `norm='ortho'` ② clamp参数 ③ 切 DoG 版 |
| FGEB 增益太小 | 中 | ① FFT→DoG→AFM 梯队 ② 增益<0.3 mAP 则降为附录 |
| FPN 蒸馏无增益 | 低 | ① 先只蒸 P3/P4 ② 调低 λ₃ |
| 审稿人质疑"和Samba区别" | 高 | ★ 表2 防撞车消融是关键 |
| 审稿人质疑"堆料" | 中 | 贡献收敛为两主轴(IG-Scan+蒸馏) + FGEB定位为轻量增强 |
| 蒸馏 Loss 压制 task Loss | 中 | ① 蒸馏 warmup ② 监控 loss 量级 |

---

## 13. 成功判据

相对 LightMamba Baseline (Row A)：
- **segm mAP ≥ +2.0**
- **AP_s ≥ +2.5**（小目标是遥感重点）
- FPS 下降 < 10%

相对大模型：
- 参数 < 25M（SAM的3-5%）
- 精度差距 < 3 mAP
- FPS 提升 ≥ 5×

---

## 14. 训练顺序（强制执行）

```
每一步只改一个变量，先确认独立有效再叠加：

1. LightMamba Baseline           → Row A
2. + FGEB                        → Row B（验证频域增强有效）
3. + FG-IG-Scan（去掉FGEB）      → Row C（验证扫描策略有效）
4. + Backbone蒸馏（去掉FGEB和IG）→ Row D
5. + Integral蒸馏(+FPN)          → Row E
6. FGEB + FG-IG-Scan             → Row F
7. 完整模型                      → Row G

如果某一步无效，不要硬叠加，先排查原因！
```

---

## 15. 一张图总结

```
┌─────────────────────────────────────────────────┐
│              RS-LightMamba 全景图                 │
├─────────────────────────────────────────────────┤
│                                                   │
│  [痛点]                                           │
│    A. 前景稀疏 → SSM 状态被背景稀释                │
│    B. 尺度差异 → 小目标 token 太少                 │
│    C. 边界模糊 + 背景高频干扰                      │
│    D. SAM→遥感域偏移                               │
│                                                   │
│  [方案]                                           │
│    ┌──────────┐  ┌───────────┐  ┌──────────────┐ │
│    │  FGEB    │→ │LightMamba │→ │FPN+MaskRCNN  │ │
│    │频域边界  │  │FG-IG-Scan │  │              │ │
│    │选择增强  │  │前景优先   │  │ 实例分割结果  │ │
│    │(C)       │  │正/逆交替  │  │              │ │
│    └──────────┘  │(A)(B)     │  └──────────────┘ │
│                  └────┬──────┘                    │
│              ┌────────┴────────┐                  │
│              │ SAM2 Integral   │ ← 只在训练时     │
│              │ Distillation    │                  │
│              │ L_edge(浅层)    │                  │
│              │ L_channel(深层) │                  │
│              │ L_pyramid(FPN)  │                  │
│              │ (D)             │                  │
│              └─────────────────┘                  │
│                                                   │
│  [指标目标]                                       │
│    参数: ~20M (SAM的3%)                            │
│    mAP: 接近大模型 (差距<3点)                      │
│    FPS: >30 (SAM的5-10倍)                          │
│    复杂度: O(N) 线性                               │
│                                                   │
│  [投稿] TGRS → 保底 JSTARS/RS                     │
└─────────────────────────────────────────────────┘
```
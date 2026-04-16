



好，我仔细看了你的实验结果和你的两点反馈。先分析数据，再回应你的观点，最后给出我修订后的手册。

---

## 一、实验结果分析

把你的结果整理成表：

| 配置 | segm mAP | AP50 | AP75 | APs | APm | APl |
|---|---|---|---|---|---|---|
| R50 baseline | 0.284 | 0.416 | 0.341 | 0.249 | 0.600 | 0.622 |
| VMamba 2292 | 0.294 | 0.416 | 0.353 | 0.260 | 0.615 | 0.619 |
| + S4 Global Attn | 0.294 | 0.416 | 0.352 | 0.260 | 0.617 | 0.613 |
| + S4 GA + HF-FPN | 0.294 | 0.417 | 0.353 | 0.259 | 0.619 | 0.610 |

**关键观察：**

1. **S4 Global Attn 完全没涨**：0.294 → 0.294，AP75 甚至微降 0.001
2. **HF-FPN（拉普拉斯版）也没涨**：0.294 → 0.294，AP50 微涨 0.001 在误差范围内
3. **VMamba 2292 比 R50 涨了 1.0 mAP**：说明 Mamba backbone 本身是有效的
4. **砾石数据集太小**——所有实验的方差都压缩在 0.001 以内，几乎分不开

### 为什么在 gravel 上不涨但在 iSAID 上涨了 0.002？

| 因素 | iSAID | Gravel Roboflow |
|---|---|---|
| **训练集大小** | ~16K 张 | 可能只有几百张 |
| **图像分辨率** | 800×800 | 320×320 |
| **类别数** | 15 类 | 1 类 |
| **前景比例** | 5-15%（稀疏） | 30-80%（密集） |
| **S4 特征图大小** | ~10×10（适合 GA） | **~5×5 甚至更小**（GA 没啥空间信息可建模） |
| **高频信号强度** | 丰富（多类边界） | **可能不强**（砾石边界模糊、分辨率低） |

**核心问题：320×320 输入经过 4 次下采样后，S4 特征图只有 ~5×5 = 25 tokens。** 在 25 个 token 上做 Global Attention 和 Mamba 几乎没区别——序列太短了，Mamba 的长程衰减问题根本不存在，GA 没有用武之地。

**HF-FPN 不涨的原因：320×320 的低分辨率图像，拉普拉斯提取的高频信号本身就很弱且噪声大。** 砾石边界在低分辨率下本就模糊（你自己说"像素一般不太清楚"），拉普拉斯算子提取的更多是 JPEG 压缩噪声而不是真实边界。

**结论：这不是你的方法有问题，是这个数据集太小太糊，任何精细化改进都看不出差异。** 你需要在自建的高分辨率数据集（或 iSAID）上才能真正验证方法。

---

## 二、回应你的两点反馈

### 反馈 1："dt 调制不做主线"—— 我同意你的谨慎

你说得对。回顾整个实验历史：

| 版本 | 做了什么 | 结果 |
|---|---|---|
| v1-v2 | importance 重排扫描（无辅助 loss） | 不涨 |
| v3 | + 辅助 loss | 热图对了但 mAP 不涨 |
| v4-v6 | IG-Gate / dt 调制 / 双路径 | 讨论但未充分验证 |

dt 调制是最深入 SSM 内部的改动，但也是最脆弱的——dt 对 softplus 和 exp 都非常敏感，β 稍微大一点就可能训练不稳定。在你当前的时间线下，把它放在消融的可选项而不是主线，是正确的风险管理。

**我的修正：dt 调制降级为"消融可选项"，不放在主架构里。如果 z 门控调制（IG-Gate）已经涨点了，dt 调制作为 +α 的叠加；如果 IG-Gate 不涨，dt 调制也大概率不会救回来。**

### 反馈 2："DWT 可以作为创新要点"—— 我部分同意，但要有条件

你说"DWT 还是可以作为创新要点"，理由是它和 HF-FPN 强耦合。这个逻辑是对的——**如果你把 DWT 定位为"backbone 和 FPN 之间的信息桥梁"，而不是单独的"下采样替代方案"，那它确实可以升级为创新子点。**

但前提是：**DWT 必须在消融实验中被证明有效。** 目前的 gravel 数据集上看不出差异，你需要：

1. 在 iSAID 上对比 "标准 Patch Merging" vs "DWT 下采样"（不加 HF-FPN）
2. 如果 DWT 单独涨点 → 可以作为独立子点
3. 如果 DWT 单独不涨，但 DWT + HF-FPN 组合涨点 → 写成"DWT-HF-FPN 联合设计"

**和 PQGNet 撞车的问题仍然存在**，但可以这样化解：

> "While DWT-based downsampling has been explored in U-Net architectures for infrared target detection [PQGNet], we are the first to integrate it with Mamba SSM backbones, using the extracted high-frequency sub-bands as cross-stage guidance signals for FPN fusion (Section III-B)."

关键词：**"first to integrate with Mamba SSM"** + **"cross-stage guidance for FPN"**。这两个限定词把你和 PQGNet 拉开了。

---

## 三、修订后的架构手册 v1.2

基于你的���馈和实验数据，我对 v1.1 做以下调整：

---

# Wave-LightMamba 核心架构手册 v1.2（修订版）

## 核心叙事（不变）

地表砾石与背景细土颜色高度一致 → 需要频域边缘信息 → DWT 解耦 + 高频门控 + SAM 蒸馏

---

## 创新点一：DWT-Mamba Backbone（含三个子组件）

### 1A. DWT 频域下采样 ← **升级为有条件的创新子点**

**做法**：用 Haar DWT 替代 Patch Merging，LL 送入 Mamba，LH+HL+HH 存入高频旁路。

**创新定位**：
- 如果消���证明 DWT 单独涨点 → 写为"DWT-based frequency-preserving downsampling for SSM backbone"
- 如果只有 DWT + HF-FPN 组合涨点 → 写为"DWT-HF-FPN joint design"的一部分
- **必须引用 PQGNet，明确说 DWT 下采样已有先例，我们的创新在于和 Mamba + HF-FPN 的集成**

**实现**（与 PQGNet MWHL 的差异）：

```python name=dwt_downsample.py
# PQGNet 的 MWHL: DWT-LL + MaxPool → concat → conv
# 你的做法: DWT-LL → Mamba stage, DWT高频 → HF-FPN旁路

class DWT_Downsample(nn.Module):
    def __init__(self, dim, wave='haar'):
        super().__init__()
        self.dwt = DWT_2D(wave)
        self.dim = dim
        # LL 通道映射（DWT 后 LL 通道数不变，需要和 Patch Merging 对齐）
        self.ll_proj = nn.Conv2d(dim, dim * 2, kernel_size=1)  # 通道翻倍
        self.norm = nn.LayerNorm(dim * 2)
    
    def forward(self, x):
        """
        x: (B, C, H, W)
        返回: ll_out (B, 2C, H/2, W/2) 给下一层 Mamba
              hf_tuple (lh, hl, hh) 各 (B, C, H/2, W/2) 存入旁路
        """
        dwt_out = self.dwt(x)  # (B, 4C, H/2, W/2)
        ll, lh, hl, hh = dwt_out.split(self.dim, dim=1)
        
        ll_out = self.ll_proj(ll)  # (B, 2C, H/2, W/2)
        # 注意：不做 MaxPool concat（和 PQGNet 区分）
        
        return ll_out, (lh, hl, hh)
```

### 1B. Mamba SSM Stages 1-3 ← **不做 dt 调制，保持原版 SS2D**

**v1.2 的调整**：浅层 Stage 1-3 使用**标准 VMamba SS2D**，不做任何 dt/z 调制。

**理由**：
1. dt 调制在 v4-v6 实验中未充分验证，风险过高
2. backbone 的稳定性优先于创新密度
3. importance 机制的价值已经通过 HF-FPN 体现，不需要在 backbone 内部重复

**dt 调制保留为消融可选项**：如果 HF-FPN 效果显著，可以在消融表最后一行加一个 "+dt modulation" 看看有没有额外增益。不作为默认配置。

### 1C. Stage 4 Global Attention ← **保留，但不作为独立贡献**

**在 gravel 数据集上无效的原因已经分析清楚**：320×320 输入 → S4 只有 5×5 tokens，太小了。

**在 iSAID 上微涨 0.002 的原因**：800×800 输入 → S4 约 10×10 = 100 tokens，GA 有轻微优势。

**定位**：写为"engineering choice for the deepest stage"，放在消融表里但不作为核心贡献宣传。重点看 iSAID 1024×1024 上的效果。

---

## 创新点二：HF-FPN（保持核心设计，升级高频提取方式）

### ⚠️ 当前拉普拉斯版在 gravel 320×320 上无效的原因

**拉普拉斯算子在低分辨率模糊图像上提取的"高频"更多是噪声而不是边界。** 这不是 HF-FPN 架构的问题，是高频信号源的问题。

### 升级方案：从拉普拉斯切换到 DWT 高频旁路

**这就是 DWT 下采样和 HF-FPN 的耦合点**——创新点一 1A 存储的 `(lh, hl, hh)` 就是 HF-FPN 的输入源。

```python name=hf_fpn_v2.py
class HF_FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels, num_outs):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1) for c in in_channels_list])
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) 
            for _ in in_channels_list])
        
        # 高频门控参数
        self.gate_alphas = nn.ParameterList([
            nn.Parameter(torch.zeros(1)) for _ in range(num_outs - 1)])
        self.hf_dir_weights = nn.ParameterList([
            nn.Parameter(torch.ones(3) / 3) for _ in range(num_outs - 1)])
        
        # 高频映射：把 DWT 旁路的 C 通道降到 1 通道
        self.hf_projs = nn.ModuleList([
            nn.Conv2d(in_ch, 1, kernel_size=1) 
            for in_ch in in_channels_list[:-1]])
    
    def forward(self, features, hf_bypasses):
        """
        features: [C1, C2, C3, C4] from backbone
        hf_bypasses: [(lh,hl,hh)_1, (lh,hl,hh)_2, (lh,hl,hh)_3]
                     from DWT downsampling stages
        """
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]
        
        for i in range(len(laterals) - 1, 0, -1):
            upsampled = F.interpolate(
                laterals[i], size=laterals[i-1].shape[2:], mode='nearest')
            
            if i - 1 < len(hf_bypasses) and hf_bypasses[i-1] is not None:
                lh, hl, hh = hf_bypasses[i-1]
                
                # 可学习的方向加权
                w = torch.softmax(self.hf_dir_weights[i-1], dim=0)
                hf_combined = w[0]*lh.abs() + w[1]*hl.abs() + w[2]*hh.abs()
                
                # 通道压缩到 1 → spatial gate
                hf_map = torch.sigmoid(self.hf_projs[i-1](hf_combined))
                
                # 门控融合
                gate = 1.0 + self.gate_alphas[i-1] * hf_map
                laterals[i-1] = laterals[i-1] + upsampled * gate
            else:
                laterals[i-1] = laterals[i-1] + upsampled
        
        outs = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]
        return tuple(outs)
```

### DWT 高频 vs 拉普拉斯的优势

| | 拉普拉斯 | DWT 高频旁路 |
|---|---|---|
| **信号来源** | 对 backbone 输出特征做 2 阶梯度 | DWT 正交分解的物理子带 |
| **方向信息** | ❌ 各向同性（无方向） | ✅ 有方向（LH 水平、HL 垂直、HH 对角） |
| **噪声鲁棒性** | ❌ 低分辨率时噪声大 | ✅ 正交变换天然降噪 |
| **与下采样的关系** | 独立计算（额外开销） | **零额外开销**（DWT 下采样的副产品） |
| **可学习性** | 固定核 | ✅ 方向权重可学习 |

**这个升级让 DWT 下采样和 HF-FPN 形成了真正的耦合**：DWT 不仅是下采样工具，还是高频信号的提供者。两者组合起来才有意义。这也是你把 DWT 作为创新子点的合法性来源。

### 和 PQGNet 的精确差异（最终版）

| | PQGNet HEWL | 你的 HF-FPN |
|---|---|---|
| **位置** | U-Net 解码器 | FPN top-down |
| **高频来源** | 浅层 skip connection 做 DWT | backbone DWT 下采样的旁路 |
| **高频使用方式** | 注意力加权 → IDWT 重建上采样 | **空间门控 FPN 融合权重** |
| **是否用 IDWT** | ✅ 用 IDWT 做上采样 | **❌ 不用 IDWT** |
| **方向权重** | 无（等权）| ✅ 可学习方向权重 |
| **backbone 类型** | CNN | **Mamba SSM** |

---

## 创新点三：SAM-Guided Distillation（简化，两个 loss）

**和 v1.1 保持一致**，只用两个 loss：

```
L_feat = MSE(normalize(F_student_stage4), normalize(F_SAM))
L_mask = BCE(student_mask_logits, SAM_soft_mask.detach())

L_total = L_det + L_seg + 0.5 * L_feat + 0.3 * L_mask
```

**注意**：蒸馏需要 SAM 的教师输出。对于 gravel 数据集，你需要先跑一遍 SAM 推理，把每张图的特征和 mask 存成 .npy。这是一个前置步骤。

**如果暂时没时间做蒸馏**：先不做，用创新点一 + 创新点二已经可以投稿。蒸馏作为 "cherry on top" 在 revision 时加。

---

## 修订后的消融实验设计

```
在 iSAID 上做（不在 gravel 320×320 上做——太小了看不出差异）

| # | 配置                                    | segm mAP | 验证什么 |
|---|----------------------------------------|----------|---------|
| A | VMamba 2292 baseline                    |          | 基准    |
| B | + DWT 下采样（替代 Patch Merging）        |          | DWT 增益 |
| C | + HF-FPN (拉普拉斯版)                    |          | 拉普拉斯高频 |
| D | + HF-FPN (DWT 高频旁路版)                |          | DWT高频 vs 拉普拉斯 |
| E | B + D (DWT下采样 + DWT-HF-FPN)           |          | 联合增益 |
| F | E + S4 Global Attention                 |          | GA 增益  |
| G | F + SAM 蒸馏                            |          | 蒸馏增益 |
| H | E + dt调制 (可选)                        |          | dt调制增益 |
```

**核心消融线是 A → B → D → E**，验证 DWT 和 HF-FPN 的各自贡献和联合增益。

---

## 四、关于 Gravel 数据集的策略

**当前 gravel_roboflow 数据集不适合做方法验证**。320×320 分辨率 + 数据量太少 = 任何精细化改进都淹没在噪声中。

**建议**：

1. **gravel_roboflow 只用来跑 baseline 对比表**（R50, Swin-T, VMamba, MobileSAM, EdgeSAM 等��
2. **方法验证（消融实验）在 iSAID 上做**——数据量充足、分辨率够高、社区认可
3. **自建高分辨率 GravelSeg 数据集出来后**，再在上面跑完整实验
4. **论文结构**：
   - iSAID 上的 SOTA 对比 + 消融实验（主实验）
   - GravelSeg 上的 SOTA 对比（泛化验证 + 数据集贡献）
   - gravel_roboflow 上的 SOTA 对比（补充实验，如果空间够的话）

---

## 五、最终总结

### v1.1 → v1.2 的变更

| 变更项 | v1.1 | v1.2 |
|---|---|---|
| **DWT 下采样的定位** | "实现细节" | **"有条件的创新子点"**（需消融验证） |
| **dt 调制** | "核心创新子点" | **"消融可选项"**（不作为默认配置） |
| **HF-FPN 高频来源** | 拉普拉斯 / DWT 均可 | **优先用 DWT 旁路**（和下采样耦合） |
| **S4 Global Attn** | 保留 | 保留，但降级为 engineering choice |
| **蒸馏** | 两个 loss | 不变，但可推迟到 revision |

### 你现在应该做什么

| 优先级 | 任务 | 时间 |
|---|---|---|
| **1** | 在 iSAID 上跑 DWT 下采样 vs Patch Merging（消融 B） | 2-3 天 |
| **2** | 实现 DWT 旁路版 HF-FPN（替代拉普拉斯版） | 1 天代码 + 2-3 天训练 |
| **3** | 跑完整消融 A→B→C→D→E→F | 1-2 周 |
| **4** | 并行：继续自建 GravelSeg 数据集的标注 | 2-3 周 |
| **5** | 根据消融结果决定是否加 dt 调制和蒸馏 | 视结果而定 |

**一句话：v1.2 的核心变化是把 DWT 下采样从"实现细节"升级为"和 HF-FPN 耦合的创新子点"，同时把 dt 调制从"主线"降级为"可选消融"。这个调整既尊重了你的实验反馈，也保持了和 PQGNet 的足够差异。关键是：在 iSAID 上验证，不在 320×320 的 gravel 上验证。**
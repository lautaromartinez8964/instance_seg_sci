import torch
import torch.nn as nn
from mamba_ssm import Mamba2
from timm.models.layers import DropPath, trunc_normal_
from functools import partial

# 层归一化
class LayerNorm2d(nn.Module):
    """2D LayerNorm (支持 [B, C, H, W] 和 [B, H, W, C] 两种格式)"""
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if x.ndim == 4:
            # 检测格式：[B, C, H, W] 或 [B, H, W, C]
            if x.size(1) == self.normalized_shape[0]:
                # [B, C, H, W] 格式，在 dim=1 归一化
                u = x.mean(1, keepdim=True)
                s = (x - u).pow(2).mean(1, keepdim=True)
                x = (x - u) / torch.sqrt(s + self.eps)
                x = self.weight[:, None, None] * x + self.bias[:, None, None]
            else:
                # [B, H, W, C] 格式，在 dim=3 归一化
                u = x.mean(-1, keepdim=True)
                s = (x - u).pow(2).mean(-1, keepdim=True)
                x = (x - u) / torch.sqrt(s + self.eps)
                x = self.weight * x + self.bias
        else:
            raise ValueError(f"期望 4D 输入，得到 {x.ndim}D")
        return x

class CrossScan_SS2D(nn.Module):
    """
    Cross-Scan 2D: VMamba 的核心创新 (高层级复现)
    对图像进行 4 个方向的扫描，捕捉全方位的空间依赖
    """
    def __init__(self, d_model, d_state=128, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        
        # 只用1个Mamba，但输入是[B, 4*d_model, L]
        self.mamba = Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)        
        # 融合层
        self.proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        """
        x: [B, H, W, C]
        """
        B, H, W, C = x.shape
        
        # 1. 生成 4 个方向的扫描序列
        x_h = x.view(B, H * W, C)
        x_h_flip = x.flip([2]).view(B, H * W, C)
        x_v = x.transpose(1, 2).contiguous().view(B, W * H, C)
        x_v_flip = x.transpose(1, 2).flip([2]).contiguous().view(B, W * H, C)
        
        # 2. 分别通过 Mamba（共享权重）
        out_h = self.mamba(x_h)
        out_h_flip = self.mamba(x_h_flip)
        out_v = self.mamba(x_v)
        out_v_flip = self.mamba(x_v_flip)
        
        # 3. 反向操作 + 相加融合（原版逻辑）
        out_h = out_h.view(B, H, W, C)
        out_h_flip = out_h_flip.view(B, H, W, C).flip([2])
        out_v = out_v.view(B, W, H, C).transpose(1, 2)
        out_v_flip = out_v_flip.view(B, W, H, C).flip([2]).transpose(1, 2)
        
        out = (out_h + out_h_flip + out_v + out_v_flip) / 4.0  # 平均融合
        return out

class VSSBlock_CrossScan(nn.Module):
    """
    带 Cross-Scan 的 VSS Block
    """
    def __init__(
        self,
        hidden_dim: int,
        drop_path: float = 0.,
        norm_layer: nn.Module = nn.LayerNorm,
        channel_first: bool = False,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        **kwargs
    ):
        super().__init__()
        self.channel_first = channel_first
        self.norm = norm_layer(hidden_dim)
        
        # 使用 Cross-Scan SS2D
        self.ss2d = CrossScan_SS2D(
            d_model=hidden_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        
        # MLP (Feed Forward Network)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(hidden_dim)
    
    def forward(self, x):
        """
        x: [B, H, W, C] if not channel_first else [B, C, H, W]
        MLP 在 SS2D 之后，先建模空间关系再增强特征
        """
        if self.channel_first:
            x = x.permute(0, 2, 3, 1).contiguous()
        
        # 分支 1: SS2D
        shortcut = x
        x = self.norm(x)
        x = self.ss2d(x)
        x = shortcut + self.drop_path(x)
        
        # 分支 2: MLP
        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut + self.drop_path(x)
        
        if self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        
        return x
    
# Patch Embedding 和 Patch Merging
class PatchEmbed(nn.Module):
    """Image to Patch Embedding
       Patch 划分：将图像分割成非重叠的小块
       特征投影：将每个 patch 映射到高维嵌入空间
       可选归一化：对嵌入特征进行标准化处理
       过程：输入 [B, C, H, W] -> 输出 [B, H/patch_size, W/patch_size, embed_dim]
       """
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        embed_dim = int(embed_dim)  # 确保 embed_dim 是整数类型
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x

class PatchMerging(nn.Module):
    """Patch Merging Layer (Downsample)
       关键下采样流程
       将2x2的相邻像素拼在一起,分辨率减半,通道数翻倍
       [B,H,W,C] -> [B,H/2,W/2,4C] -> [B,H/2,W/2,2C]

    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        dim = int(dim)  # 确保 dim 是整数类型
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = self.norm(x)
        x = self.reduction(x)
        return x
        
class VSSLayer(nn.Module):
    """
    一个 Stage，包含多个 VSSBlock_CrossScan
    【修复】增加了在 downsample 前后的维度转换
    """
    def __init__(
        self,
        dim,
        depth,
        drop_path=0.,
        norm_layer=nn.LayerNorm,
        downsample=None,
        channel_first=False,
        **kwargs
    ):
        super().__init__()
        self.channel_first = channel_first # 🔥 必须记住这个标记
        self.blocks = nn.ModuleList([
            VSSBlock_CrossScan(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                channel_first=channel_first,
                **kwargs
            )
            for i in range(depth)
        ])
        
        # 🔥 关键修复：实例化 downsample 而不是直接保存类
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        # 1. 执行 Block (Block 内部会自动处理 channel_first，所以这里不用管)
        for blk in self.blocks:
            x = blk(x)
        
        # 2. 执行 Downsample (关键修复点！)
        if self.downsample is not None:
            if self.channel_first:
                # 如果当前是 [B, C, H, W]，必须转成 [B, H, W, C] 才能进 PatchMerging
                x_down = x.permute(0, 2, 3, 1).contiguous()
                x_down = self.downsample(x_down)
                # 出来后再转回 [B, 2C, H/2, W/2]
                x_down = x_down.permute(0, 3, 1, 2).contiguous()
                return x, x_down
            else:
                # 如果本来就是 [B, H, W, C]，直接进
                x_down = self.downsample(x)
                return x, x_down
        else:
            return x, x
        
class VSSM_Official(nn.Module):
    """
    官方 Mamba-SSM 版本的 VMamba (主干网络)
    【修复】forward 中增加了 PatchEmbed 后的维度对齐
    """
    def __init__(
        self,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        depths=[2, 2, 9, 2],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.2,
        norm_layer='ln',
        channel_first=False,
        **kwargs
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.dims = dims
        self.channel_first = channel_first
        
        if norm_layer == 'ln': norm_layer = nn.LayerNorm
        elif norm_layer == 'ln2d': norm_layer = LayerNorm2d; self.channel_first = True
        elif norm_layer == 'bn': norm_layer = nn.BatchNorm2d; self.channel_first = True
        
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=dims[0],
            norm_layer=norm_layer if norm_layer != nn.LayerNorm else None
        )
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if i_layer < self.num_layers - 1 else None,
                # 🔥 这一行非常重要，要把 channel_first 传给 Layer
                channel_first=self.channel_first, 
                **kwargs
            )
            self.layers.append(layer)
        
        self.head = nn.Linear(dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        # x: [B, 3, H, W]
        x = self.patch_embed(x) # 输出是 [B, H/4, W/4, C] (Channel Last)
        
        # 🔥 关键修复：如果你用 ln2d (Channel First)，必须在这里转置！
        if self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous() # 转成 [B, C, H, W]
            
        # 🔥 使用 downsample 的输出作为下一层的输入
        for layer in self.layers:
            _, x = layer(x)  # 第二个返回值是 downsample 后的输出
            
        # 结尾处理：Global Average Pooling
        if self.channel_first:
            # [B, C, H, W] -> [B, C]
            x = x.mean([2, 3])
        else:
            # [B, H, W, C] -> [B, C]
            x = x.mean([1, 2])
            
        x = self.head(x)
        return x
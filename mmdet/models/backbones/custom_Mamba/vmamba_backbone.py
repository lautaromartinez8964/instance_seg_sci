import torch
import torch.nn as nn
from mmdet.registry import MODELS
from .vmamba import VSSM_Official, VSSBlock_CrossScan as VSSBlock_Official, LayerNorm2d

# changemamba_backbone:将通用的VMamba分类模型无缝转换为变化检测任务的特征提取器
# 多尺度输出接口：为不同层级添加归一化层
# 分类器移除：删除不需要的分类头
# 权重加载：支持预训练权重
# 冻结控制：灵活的层冻结策略


@MODELS.register_module()
class VMambaBackbone(VSSM_Official):
    """
    ChangeMamba 的 Backbone (基于官方 Mamba-SSM)
    适配 OpenCD 的 Backbone 接口
    """
    def __init__(
        self,
        out_indices=(0, 1, 2, 3), # 关键1：多尺度输出接口 0：浅层特征 3：最深层，最小分辨率
        pretrained=None,
        frozen_stages=-1,
        norm_layer='ln2d',
        **kwargs
    ):
        super().__init__(norm_layer=norm_layer, **kwargs)
        
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        
        # 关键2：归一化层的动态添加
        # 为每个输出层添加 norm (OpenCD 规范)
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer_cls = _NORMLAYERS.get(norm_layer.lower(), nn.LayerNorm)
        
        for i in out_indices:
            layer = norm_layer_cls(self.dims[i]) # 维度自适应
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)
        
        # 删除分类头 (我们只要特征)
        del self.head
        if hasattr(self, 'avgpool'):
            del self.avgpool
        
        # 关键3：加载预训练权重
        if pretrained: 
            self.load_pretrained(pretrained)
        
        # 关键5：冻结指定层 这里为0，1，2，-1（冻结stage0， stage0，1 ，stage0，1，2； -1为解冻）
        self._freeze_stages()
    
    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
        
        for i in range(self.frozen_stages):
            m = self.layers[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
    
    def load_pretrained(self, ckpt_path):
        """加载预训练权重"""
        try:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # 过滤掉分类头的权重
            state_dict = {k: v for k, v in state_dict.items() 
                         if not k.startswith('head')}
            
            msg = self.load_state_dict(state_dict, strict=False)
            print(f"✅ Loaded pretrained weights from {ckpt_path}")
            print(f"Missing keys: {msg.missing_keys}")
            print(f"Unexpected keys: {msg.unexpected_keys}")
        except Exception as e:
            print(f"⚠️ Failed to load pretrained weights: {e}")
    
    def forward(self, x):
        """
        输入: x [B, 3, H, W]
        输出: list of features [feat0, feat1, feat2, feat3]
        """
        x = self.patch_embed(x)  # [B, H//4, W//4, C0]
        
        # 🔥 修复：如果使用 ln2d (channel_first=True)，需要在这里转置维度
        if self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        
        outs = []
        for i, layer in enumerate(self.layers):
            x_out, x = layer(x)  # x_out:  当前层输出, x: 传给下一层
            
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(x_out)
                
                # 确保输出是 [B, C, H, W] 格式
                if not self.channel_first:
                    out = out.permute(0, 3, 1, 2).contiguous()
                
                outs.append(out)
        
        return outs

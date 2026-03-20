import torch
import torch.nn as nn
from mmdet.registry import MODELS
from .vmamba import Backbone_VSSM

@MODELS.register_module()
class MM_VMamba(Backbone_VSSM):
    """
    单继承的极简 Wrapper：直接继承官方主干，剔除干扰参数。
    """
    def __init__(self, *args, **kwargs):
        # 拦截并删除 MMDetection 可能自动传入但不被 VMamba 识别的参数
        kwargs.pop('init_cfg', None)
        
        # 强制指定输出 4 个 Stage (对于检测任务必须)
        kwargs['out_indices'] = (0, 1, 2, 3)
        
        super().__init__(*args, **kwargs)

    def init_weights(self):
        """留空即可，因为我们会在 config 里用 pretrained 参数加载权重"""
        pass
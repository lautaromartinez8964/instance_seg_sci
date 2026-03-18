import torch
import torch.nn as nn
from mmdet.registry import MODELS
from mmengine.model import BaseModule

# 导入官方刚刚搬过来的 vmamba 里面的 Backbone_VSSM
from .vmamba import Backbone_VSSM

@MODELS.register_module()
class MM_VMamba(BaseModule, Backbone_VSSM):
    """
    将官方的 Backbone_VSSM 包装成 MMDetection 认识的 Backbone
    """
    def __init__(self, *args, **kwargs):
        BaseModule.__init__(self)
        Backbone_VSSM.__init__(self, *args, **kwargs)
        
    def init_weights(self):
        """MMDetection的权重初始化规范接口，官方源码自带的 load_pretrained 已经处理过了，这里设为 pass"""
        pass
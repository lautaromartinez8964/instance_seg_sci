import traceback

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
from mmdet.registry import MODELS

# 显式导入双轨 Backbone
from mmdet.models.backbones.vmamba_official.mmdet_vssm import MM_VMamba
from mmdet.models.backbones.custom_Mamba.vmamba_backbone import VMambaBackbone

print("🚀 初始化测试环境...\n")
dummy_input = torch.randn(2, 3, 1024, 1024).cuda()

# ==========================================================
# 轨道 A：测试官方底层极速版
# ==========================================================
print("========== [轨道 A] 测试 官方 VMamba (MM_VMamba) ==========")
official_cfg = dict(
    type='MM_VMamba',
    depths=[2, 2, 9, 2],       
    dims=[96, 192, 384, 768],  
    drop_path_rate=0.2,
)

try:
    model_official = MODELS.build(official_cfg).cuda().eval()
    with torch.no_grad():
        out_official = model_official(dummy_input)
    print("✅ 官方版前向传播成功！输出特征如下：")
    for i, out in enumerate(out_official):
        print(f"   Stage {i+1} shape: {out.shape}")
except Exception as e:
    print("\n❌ 官方版失败，真正的详细报错如下:")
    traceback.print_exc()  # 这行会把底层的红字全部扒出来！

print("\n" + "="*60 + "\n")

# ==========================================================
# 轨道 B：测试你的灵活魔改版
# ==========================================================
print("========== [轨道 B] 测试 自定义 VMamba (VMambaBackbone) ==========")
custom_cfg = dict(
    type='VMambaBackbone',
    depths=[2, 2, 9, 2],       
    dims=[96, 192, 384, 768],
    out_indices=(0, 1, 2, 3),
    norm_layer='ln2d'
)

try:
    model_custom = MODELS.build(custom_cfg).cuda().eval()
    with torch.no_grad():
        out_custom = model_custom(dummy_input)
    print("✅ 自定义版前向传播成功！输出特征如下：")
    for i, out in enumerate(out_custom):
        print(f"   Stage {i+1} shape: {out.shape}")
except Exception as e:
    print(f"❌ 自定义版失败: {e}")

print("\n" + "="*60 + "\n")
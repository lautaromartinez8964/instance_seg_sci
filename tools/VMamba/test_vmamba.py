import torch
from mmdet.registry import MODELS
import mmdet.models

# 假设你在 MMDetection 里注册的名字叫 'VMamba'
backbone_cfg = dict(
    type='MM_VMamba',
    arch='tiny', # 根据你代码里的实际参数调整
    out_indices=(0, 1, 2, 3),
    drop_path_rate=0.2,
)

print("正在构建 VMamba Backbone...")
model = MODELS.build(backbone_cfg).cuda()
model.eval()

print("生成模拟遥感图像[Batch=2, Channel=3, H=1024, W=1024]...")
dummy_input = torch.randn(2, 3, 1024, 1024).cuda()

print("开始前向传播...")
with torch.no_grad():
    outputs = model(dummy_input)

print("\n========== VMamba 输出特征维度 ==========")
for i, out in enumerate(outputs):
    print(f"Stage {i+1} 特征维度: {out.shape}")
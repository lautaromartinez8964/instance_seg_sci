import torch
from pathlib import Path

src = "work_dirs/mask_rcnn_r50_fpn_1x_isaid_noval_resume/epoch_11.pth"
dst = "work_dirs/mask_rcnn_r50_fpn_1x_isaid_noval_resume/epoch_11_weights_only.pth"

# 你自己的checkpoint，可信任，允许完整反序列化
ckpt = torch.load(src, map_location="cpu", weights_only=False)

# 只保留模型权重
if isinstance(ckpt, dict) and "state_dict" in ckpt:
    out = {"state_dict": ckpt["state_dict"]}
else:
    out = {"state_dict": ckpt}

torch.save(out, dst)
print(f"saved: {dst}")
print(f"keys: {list(out.keys())}")
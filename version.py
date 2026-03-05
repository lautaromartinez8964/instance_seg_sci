import sys
print(f"========== 🐍 Python 环境 ==========")
print(f"Python 版本: {sys.version.split()[0]}")

print(f"\n========== 🔥 PyTorch & CUDA 环境 ==========")
try:
    import torch
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"PyTorch 绑定的 CUDA 版本: {torch.version.cuda}")
        print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("❌ PyTorch 未安装！")

print(f"\n========== 📦 OpenMMLab 体系 ==========")
try:
    import mmdet
    print(f"MMDetection 版本: {mmdet.__version__}")
except ImportError:
    print("❌ MMDetection 未安装！")

try:
    import mmcv
    print(f"MMCV 版本: {mmcv.__version__}")
except ImportError:
    print("❌ MMCV 未安装！(注意: mmdet强依赖mmcv)")

try:
    import mmengine
    print(f"MMEngine 版本: {mmengine.__version__}")
except ImportError:
    print("❌ MMEngine 未安装！")

print(f"\n========== 🧬 Mamba 核心依赖 ==========")
try:
    import mamba_ssm
    print(f"mamba_ssm 版本: {mamba_ssm.__version__}")
except ImportError:
    print("⚠️ mamba_ssm 未安装！(暂时没关系，后面用到再装)")

try:
    import causal_conv1d
    print(f"causal_conv1d 版本: {causal_conv1d.__version__}")
except ImportError:
    print("⚠️ causal_conv1d 未安装！")

try:
    import transformers
    print(f"Transformers 版本: {transformers.__version__}")
except ImportError:
    print("⚠️ Transformers 未安装！")

try:
    import timm
    print(f"TIMM 版本: {timm.__version__}")
except ImportError:
    print("⚠️ TIMM 未安装！")


print("\n========================================")
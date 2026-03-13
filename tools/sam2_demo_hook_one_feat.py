import cv2
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor

feat_cache = {}

def make_hook(name):
    def _hook(module, inputs, outputs):
        if torch.is_tensor(outputs):
            feat_cache[name] = outputs.detach().cpu()
        elif isinstance(outputs, (list, tuple)):
            ts = [x.detach().cpu() for x in outputs if torch.is_tensor(x)]
            if len(ts) > 0:
                feat_cache[name] = ts
    return _hook

def main():
    predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
    model = predictor.model

    # 1) 定位 image_encoder
    image_encoder = getattr(model, "image_encoder", None)
    if image_encoder is None:
        for n, m in model.named_modules():
            if "image_encoder" in n:
                image_encoder = m
                break
    assert image_encoder is not None, "没找到 image_encoder，请打印 model 看结构"

    # 2) 自动找一个“叶子层”挂 hook
    target_name, target_module = None, None
    for n, m in image_encoder.named_modules():
        if n == "":
            continue
        if len(list(m.children())) == 0:   # 叶子模块
            target_name, target_module = f"image_encoder.{n}", m
            break

    assert target_module is not None, "没找到可hook的子层"
    target_module.register_forward_hook(make_hook(target_name))
    print(f"[Hook] registered on: {target_name}")

    # 3) 触发前向（set_image 会走 encoder）
    img = cv2.imread("demo/demo.jpg")
    assert img is not None, "读图失败: demo/demo.jpg"
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with torch.inference_mode():
        predictor.set_image(img)

    # 4) 打印至少1层特征shape
    if len(feat_cache) == 0:
        print("没有抓到特征，请把 model 结构贴给我，我帮你精确指定层名。")
    else:
        print("\n=== Captured feature shapes ===")
        for k, v in feat_cache.items():
            if torch.is_tensor(v):
                print(f"{k}: {tuple(v.shape)}")
            else:
                print(f"{k}: {[tuple(t.shape) for t in v]}")

        # 可选：保存到磁盘
        torch.save(feat_cache, "demo/sam2_one_layer_feat.pt")
        print("saved: demo/sam2_one_layer_feat.pt")

if __name__ == "__main__":
    main()
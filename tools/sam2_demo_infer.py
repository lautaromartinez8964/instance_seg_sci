import os
import numpy as np
import torch
import cv2

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor

def overlay_mask(image_rgb, mask, color=(30, 144, 255), alpha=0.45):
    """image_rgb: HWC uint8, mask: HW bool/0-1"""
    out = image_rgb.copy().astype(np.float32)
    color_arr = np.array(color, dtype=np.float32)
    m = mask.astype(bool)
    out[m] = out[m] * (1 - alpha) + color_arr * alpha
    return out.astype(np.uint8)

def main():
    os.makedirs("demo/sam2_vis", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
    mask_generator = SAM2AutomaticMaskGenerator(
        model=predictor.model,
        points_per_side=32,
        pred_iou_thresh=0.8,
        stability_score_thresh=0.95,
        min_mask_region_area=100,
    )

    img_path = "/home/yxy18034962/projects/mmdetection/data/iSAID/train/images/P0005_0_0_800_800.png"
    bgr = cv2.imread(img_path)
    assert bgr is not None, f"read image failed: {img_path}"
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    with torch.inference_mode():
        anns = mask_generator.generate(image)

    print("num auto masks:", len(anns))
    if len(anns) == 0:
        print("No masks generated.")
        return

    anns = sorted(anns, key=lambda x: float(x.get("predicted_iou", 0.0)), reverse=True)

    # 叠加前20个mask，输出总览图
    merged = image.copy().astype(np.uint8)
    top_k_merge = min(20, len(anns))
    rng = np.random.default_rng(2026)
    for i in range(top_k_merge):
        m = anns[i]["segmentation"]
        color = tuple(int(c) for c in rng.integers(0, 256, size=3))
        merged = overlay_mask(merged, m, color=color, alpha=0.35)

    merged_path = "demo/sam2_vis/auto_masks_merged_top20.png"
    cv2.imwrite(merged_path, cv2.cvtColor(merged, cv2.COLOR_RGB2BGR))
    print(f"[Saved] {merged_path}")

    # 单独保存前10个高分mask，便于检查
    top_k_single = min(10, len(anns))
    for i in range(top_k_single):
        ann = anns[i]
        m = ann["segmentation"]
        score = float(ann.get("predicted_iou", 0.0))
        area = int(ann.get("area", int(np.sum(m))))
        vis = overlay_mask(image, m, color=(255, 99, 71), alpha=0.45)
        save_path = f"demo/sam2_vis/auto_mask_{i:02d}_iou_{score:.4f}_area_{area}.png"
        cv2.imwrite(save_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print(f"[Saved] {save_path}")

if __name__ == "__main__":
    main()
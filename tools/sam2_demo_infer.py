import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

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

    img_path = "demo/demo.jpg"
    bgr = cv2.imread(img_path)
    assert bgr is not None, f"read image failed: {img_path}"
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    h, w = image.shape[:2]
    point = np.array([[w // 2, h // 2]])
    label = np.array([1])

    with torch.inference_mode():
        predictor.set_image(image)
        masks, scores, logits = predictor.predict(
            point_coords=point,
            point_labels=label,
            multimask_output=True
        )

    print("masks shape :", np.array(masks).shape)
    print("scores shape:", np.array(scores).shape)
    print("logits shape:", np.array(logits).shape)

    # 保存每个候选mask叠加图
    for i, (m, s) in enumerate(zip(masks, scores)):
        vis = overlay_mask(image, m > 0, color=(255, 99, 71), alpha=0.45)

        # 画提示点
        x, y = point[0]
        cv2.circle(vis, (int(x), int(y)), 6, (0, 255, 0), -1)  # 注意vis是RGB显示逻辑没问题

        save_path = f"demo/sam2_vis/mask_{i}_score_{float(s):.4f}.png"
        cv2.imwrite(save_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print(f"[Saved] {save_path}")

    # 另外保存“最高分mask”
    best_idx = int(np.argmax(scores))
    best_vis = overlay_mask(image, masks[best_idx] > 0, color=(30, 144, 255), alpha=0.5)
    best_path = "demo/sam2_vis/best_mask.png"
    cv2.imwrite(best_path, cv2.cvtColor(best_vis, cv2.COLOR_RGB2BGR))
    print(f"[Saved] {best_path}, best_idx={best_idx}, score={float(scores[best_idx]):.4f}")

if __name__ == "__main__":
    main()
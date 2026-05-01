"""
SAM2 离线监督信号生成脚本 (UAGD v2.1)
=====================================
生成三个监督信号:
  - D_tea  : 距离变换图 (median across K runs), shape (N_inst, H, W), float16
  - B_tea  : 边界频率图 (boundary frequency), shape (N_inst, H, W), float16
  - W_unc  : 实例级不确定性权重,           shape (N_inst,),       float32
  - inst_ids: GT annotation id 列表,       shape (N_inst,),       int32

输出: data/gravel_big/{split}_sam2/  中每张图对应一个 .npz 文件

Usage:
    conda run -n mmdec python tools/generate_sam2_supervision.py \
        --split train \
        --num-images 10          # 调试时只处理前N张
    conda run -n mmdec python tools/generate_sam2_supervision.py \
        --split train            # 全量
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

# ── SAM2 ──────────────────────────────────────────────────────────────────────
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    print("[ERROR] sam2 not found. Run in mmdec conda env.", file=sys.stderr)
    sys.exit(1)

import torch

# ── Constants ─────────────────────────────────────────────────────────────────
SAM2_CKPT = (
    "/home/yxy18034962/.cache/huggingface/hub/"
    "models--facebook--sam2-hiera-large/snapshots/"
    "e6a8e8809b8f1bfa2238b6d080f3d05cc76bd251/sam2_hiera_large.pt"
)
SAM2_CFG = "sam2_hiera_l.yaml"  # resolves relative to sam2 package root
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# UAGD v2.1 hyperparams
K_RUNS = 5          # SAM2 推理次数
BOX_JITTER = 2      # GT bbox 微扰像素 (相对 point jitter 更温和, 不跑到邻居去)
# NOTE: box prompt 代替 point prompt, 解决密集场景漏检问题


# ── Helpers ───────────────────────────────────────────────────────────────────

def poly_to_mask(polygon_flat, H, W):
    """将 COCO polygon (flat list) 转为 bool mask."""
    pts = np.array(polygon_flat, dtype=np.float32).reshape(-1, 2)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
    return mask.astype(bool)


def mask_to_boundary(mask, thickness=2):
    """提取 mask 的边界 (dilate XOR原始)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness * 2 + 1, thickness * 2 + 1))
    dilated = cv2.dilate(mask.astype(np.uint8), kernel)
    boundary = dilated - mask.astype(np.uint8)
    return boundary.astype(bool)


def best_mask_by_iou(masks, gt_mask):
    """从 SAM2 返回的 N 个 mask 中选 IoU 最高的."""
    best_iou, best_idx = -1.0, 0
    for i, m in enumerate(masks):
        inter = (m & gt_mask).sum()
        union = (m | gt_mask).sum()
        iou = inter / (union + 1e-6)
        if iou > best_iou:
            best_iou, best_idx = iou, i
    return masks[best_idx]


def compute_dt(mask):
    """计算 mask 内每个像素到边界的欧式距离 (0在外部/边界, 正值在内部)."""
    return distance_transform_edt(mask)


def compute_instance_supervision(predictor, image_rgb, gt_mask, gt_bbox, rng):
    """
    对单个 GT 实例运行 K_RUNS 次 SAM2 (使用 Box Prompt), 返回:
        dt_stack   : (K_RUNS, H, W) float32
        bnd_stack  : (K_RUNS, H, W) bool
        best_masks : (K_RUNS, H, W) bool

    Box prompt 相比 point prompt 的优势：
      - 明确限定空间范围, 避免密集场景中跑到邻居石头上
      - 砾石平均92.6个/图, point prompt 质心极易落入相邻实例区域
    """
    H, W = gt_mask.shape
    ys, xs = np.where(gt_mask)
    if len(ys) == 0:
        zeros = np.zeros((K_RUNS, H, W), dtype=np.float32)
        return zeros, zeros.astype(bool), zeros.astype(bool)

    # GT bbox [x, y, w, h] → [x1, y1, x2, y2]
    bx, by, bw, bh = gt_bbox
    x1_gt, y1_gt = float(bx), float(by)
    x2_gt, y2_gt = float(bx + bw), float(by + bh)

    dt_stack = np.zeros((K_RUNS, H, W), dtype=np.float32)
    bnd_stack = np.zeros((K_RUNS, H, W), dtype=bool)
    best_masks = np.zeros((K_RUNS, H, W), dtype=bool)

    for k in range(K_RUNS):
        # bbox 微扰 (±BOX_JITTER px), 仍在图像范围内
        jx1 = float(np.clip(x1_gt + rng.uniform(-BOX_JITTER, BOX_JITTER), 0, W - 1))
        jy1 = float(np.clip(y1_gt + rng.uniform(-BOX_JITTER, BOX_JITTER), 0, H - 1))
        jx2 = float(np.clip(x2_gt + rng.uniform(-BOX_JITTER, BOX_JITTER), jx1 + 1, W))
        jy2 = float(np.clip(y2_gt + rng.uniform(-BOX_JITTER, BOX_JITTER), jy1 + 1, H))
        box = np.array([jx1, jy1, jx2, jy2], dtype=np.float32)

        with torch.inference_mode():
            masks_logits, scores, _ = predictor.predict(
                box=box,
                multimask_output=False,   # box prompt → SAM2 直接给最优 mask
            )
        pred_masks = masks_logits > 0.0   # (1, H, W) bool
        chosen = pred_masks[0]
        best_masks[k] = chosen
        dt_stack[k] = compute_dt(chosen).astype(np.float32)
        bnd_stack[k] = mask_to_boundary(chosen)

    return dt_stack, bnd_stack, best_masks


def compute_image_supervision(predictor, image_rgb, annotations, rng):
    """
    处理一张图片的所有 annotation, 返回:
        D_tea   : (N_inst, H, W) float16
        B_tea   : (N_inst, H, W) float16
        W_unc   : (N_inst,)      float32
        inst_ids: (N_inst,)      int32
    """
    H, W = image_rgb.shape[:2]
    N = len(annotations)

    D_tea = np.zeros((N, H, W), dtype=np.float16)
    B_tea = np.zeros((N, H, W), dtype=np.float16)
    W_unc = np.zeros(N, dtype=np.float32)
    inst_ids = np.zeros(N, dtype=np.int32)

    # 设置图像 (只调用一次, SAM2 encode)
    predictor.set_image(image_rgb)

    for i, ann in enumerate(annotations):
        inst_ids[i] = ann['id']
        seg = ann.get('segmentation', [])
        if not seg or ann.get('iscrowd', 0):
            # iscrowd 或无分割, 跳过, 权重=0
            continue

        # 取第一个多边形 (COCO 格式可能有多个)
        gt_mask = poly_to_mask(seg[0], H, W)
        gt_bbox = ann.get('bbox', [0, 0, W, H])   # [x, y, w, h]

        dt_stack, bnd_stack, best_masks = compute_instance_supervision(
            predictor, image_rgb, gt_mask, gt_bbox, rng
        )

        # ── D_tea: K 次推理 DT 的中值 ──────────────────────────────────────
        # 用等效半径归一化 (保持大小砾石等权)
        area = gt_mask.sum()
        r_equiv = np.sqrt(area / np.pi) + 1e-3
        d_median = np.median(dt_stack, axis=0) / r_equiv   # (H, W)
        D_tea[i] = d_median.astype(np.float16)

        # ── B_tea: 边界频率图 ────────────────────────────────────────────────
        b_freq = bnd_stack.mean(axis=0)   # (H, W) float64 → [0,1]
        B_tea[i] = b_freq.astype(np.float16)

        # ── W_unc: 实例级不确定性权重 ────────────────────────────────────────
        # W_unc = 1 - mean_pixel_variance  (稳定预测 → 权重高)
        # best_masks: (K_RUNS, H, W) bool
        pixel_var = best_masks.var(axis=0)       # (H, W)
        mean_var = pixel_var[gt_mask].mean() if gt_mask.sum() > 0 else 0.0
        W_unc[i] = float(np.clip(1.0 - mean_var, 0.0, 1.0))

    return D_tea, B_tea, W_unc, inst_ids


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Generate SAM2 supervision signals")
    parser.add_argument("--split", default="train", choices=["train", "valid", "test"])
    parser.add_argument("--data-root", default="data/gravel_big")
    parser.add_argument("--num-images", type=int, default=None,
                        help="Only process first N images (for debugging)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true",
                        help="Skip images that already have .npz output")
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    data_root = Path(args.data_root)
    split_dir = data_root / args.split
    ann_file = split_dir / "_annotations.coco.json"
    out_dir = data_root / f"{args.split}_sam2"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading annotations from {ann_file}")
    with open(ann_file) as f:
        coco = json.load(f)

    # 建立 image_id → annotations 的映射
    ann_by_img = {}
    for ann in coco['annotations']:
        ann_by_img.setdefault(ann['image_id'], []).append(ann)

    images = coco['images']
    if args.num_images is not None:
        images = images[:args.num_images]
        print(f"[INFO] DEBUG mode: processing first {args.num_images} images")

    # ── 加载 SAM2 ─────────────────────────────────────────────────────────────
    print(f"[INFO] Loading SAM2 on {DEVICE} ...")
    sam2_model = build_sam2(SAM2_CFG, SAM2_CKPT, device=DEVICE)
    predictor = SAM2ImagePredictor(sam2_model)
    print("[INFO] SAM2 loaded.")

    # ── 主循环 ────────────────────────────────────────────────────────────────
    skipped = 0
    for img_info in tqdm(images, desc=f"Generating SAM2 supervision [{args.split}]"):
        img_id = img_info['id']
        img_fname = img_info['file_name']
        out_path = out_dir / (Path(img_fname).stem + ".npz")

        if args.resume and out_path.exists():
            skipped += 1
            continue

        img_path = split_dir / img_fname
        if not img_path.exists():
            print(f"[WARN] Image not found: {img_path}", file=sys.stderr)
            continue

        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"[WARN] Failed to read: {img_path}", file=sys.stderr)
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        annotations = ann_by_img.get(img_id, [])
        if not annotations:
            continue

        D_tea, B_tea, W_unc, inst_ids = compute_image_supervision(
            predictor, image_rgb, annotations, rng
        )

        np.savez_compressed(
            out_path,
            D_tea=D_tea,      # (N_inst, H, W) float16
            B_tea=B_tea,      # (N_inst, H, W) float16
            W_unc=W_unc,      # (N_inst,)      float32
            inst_ids=inst_ids # (N_inst,)      int32
        )

    print(f"\n[DONE] Saved to {out_dir}  (skipped {skipped} existing files)")
    print(f"       Total files: {len(list(out_dir.glob('*.npz')))}")


if __name__ == "__main__":
    main()

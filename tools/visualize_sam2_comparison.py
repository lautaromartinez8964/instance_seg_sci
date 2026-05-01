"""
SAM2-Large vs SAM2-Small 分割质量对比可视化脚本 (UAGD v2.1)
=============================================================
对随机选取的 N 张图片，展示：
  面板 A：原图 + GT 轮廓
  面板 B：SAM2-Large  B_tea composite 边界置信度图
  面板 C：SAM2-Small  B_tea composite 边界置信度图
  面板 D：W_unc 不确定性权重分布（per-instance 热力图叠加在图上）
  面板 E：Large vs Small 差异图 (abs diff)

用法（建议 mmdec conda 环境）：
    conda run -n mmdec python tools/visualize_sam2_comparison.py \
        --num-images 25 --seed 42
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')   # 无头模式
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    print("[ERROR] sam2 not found. Run in mmdec conda env.", file=sys.stderr)
    sys.exit(1)

import torch

# ── Model paths ───────────────────────────────────────────────────────────────
_SNAP_ROOT = Path("/home/yxy18034962/.cache/huggingface/hub")

SAM2_LARGE_CKPT = str(
    _SNAP_ROOT
    / "models--facebook--sam2-hiera-large/snapshots"
    / "e6a8e8809b8f1bfa2238b6d080f3d05cc76bd251/sam2_hiera_large.pt"
)
SAM2_SMALL_CKPT = str(
    _SNAP_ROOT
    / "models--facebook--sam2-hiera-small/snapshots"
    / "e080ada8afd19df5e165abe71b006edc7f4c3d4e/sam2_hiera_small.pt"
)
SAM2_LARGE_CFG = "sam2_hiera_l.yaml"
SAM2_SMALL_CFG = "sam2_hiera_s.yaml"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K_RUNS = 5
BOX_JITTER = 2      # GT bbox 微扰像素, 武器比 point jitter 温和且不跑到邻居去


# ── Helpers (同 generate 脚本) ────────────────────────────────────────────────

def poly_to_mask(polygon_flat, H, W):
    pts = np.array(polygon_flat, dtype=np.float32).reshape(-1, 2)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
    return mask.astype(bool)


def mask_to_boundary(mask, thickness=2):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (thickness * 2 + 1, thickness * 2 + 1))
    dilated = cv2.dilate(mask.astype(np.uint8), kernel)
    return (dilated - mask.astype(np.uint8)).astype(bool)


def best_mask_by_iou(masks, gt_mask):
    best_iou, best_idx = -1.0, 0
    for i, m in enumerate(masks):
        inter = (m & gt_mask).sum()
        union = (m | gt_mask).sum()
        iou = inter / (union + 1e-6)
        if iou > best_iou:
            best_iou, best_idx = iou, i
    return masks[best_idx]


def run_sam2_on_image(predictor, image_rgb, annotations, rng):
    """
    使用 Box Prompt 运行 SAM2, 返回:
        B_comp   : (H, W) float32  ─ W_unc 加权最大聚合边界置信度图
        W_unc    : (N_inst,) float32
        gt_masks : list of (H,W) bool

    Box Prompt vs Point Prompt:
      - Point: 质心坐标极易落入相邻石头 → 大量漏检
      - Box  : GT bbox 限定搜索范围 → 每实例禄葛垣分割准确
    """
    H, W = image_rgb.shape[:2]
    predictor.set_image(image_rgb)

    B_tea_list = []
    W_unc_list = []
    gt_masks = []

    for ann in annotations:
        seg = ann.get('segmentation', [])
        if not seg or ann.get('iscrowd', 0):
            continue
        gt_mask = poly_to_mask(seg[0], H, W)
        gt_masks.append(gt_mask)

        # GT bbox [x, y, w, h] → [x1, y1, x2, y2]
        bx, by, bw, bh = ann.get('bbox', [0, 0, W, H])
        x1_gt, y1_gt = float(bx), float(by)
        x2_gt, y2_gt = float(bx + bw), float(by + bh)

        bnd_stack = np.zeros((K_RUNS, H, W), dtype=bool)
        best_masks_k = np.zeros((K_RUNS, H, W), dtype=bool)

        ys, xs = np.where(gt_mask)
        if len(ys) == 0:
            B_tea_list.append(np.zeros((H, W), dtype=np.float32))
            W_unc_list.append(0.0)
            continue

        for k in range(K_RUNS):
            # bbox 微扰 ±BOX_JITTER px
            jx1 = float(np.clip(x1_gt + rng.uniform(-BOX_JITTER, BOX_JITTER), 0, W - 1))
            jy1 = float(np.clip(y1_gt + rng.uniform(-BOX_JITTER, BOX_JITTER), 0, H - 1))
            jx2 = float(np.clip(x2_gt + rng.uniform(-BOX_JITTER, BOX_JITTER), jx1 + 1, W))
            jy2 = float(np.clip(y2_gt + rng.uniform(-BOX_JITTER, BOX_JITTER), jy1 + 1, H))
            box = np.array([jx1, jy1, jx2, jy2], dtype=np.float32)

            with torch.inference_mode():
                masks_logits, _, _ = predictor.predict(
                    box=box,
                    multimask_output=False)   # box prompt 直接给最优 mask
            chosen = (masks_logits > 0.0)[0]   # (H, W) bool
            best_masks_k[k] = chosen
            bnd_stack[k] = mask_to_boundary(chosen)

        b_freq = bnd_stack.mean(axis=0).astype(np.float32)
        B_tea_list.append(b_freq)

        pixel_var = best_masks_k.var(axis=0)
        mean_var = pixel_var[gt_mask].mean() if gt_mask.sum() > 0 else 0.0
        W_unc_list.append(float(np.clip(1.0 - mean_var, 0.0, 1.0)))

    if not B_tea_list:
        return (np.zeros((H, W), dtype=np.float32),
                np.array([], dtype=np.float32),
                gt_masks)

    B_tea_arr = np.stack(B_tea_list, axis=0)           # (N, H, W)
    W_unc_arr = np.array(W_unc_list, dtype=np.float32) # (N,)
    # W_unc 加权最大聚合
    B_comp = (W_unc_arr[:, None, None] * B_tea_arr).max(axis=0)
    return B_comp, W_unc_arr, gt_masks


def draw_gt_contours(image_rgb, gt_masks, color=(0, 255, 0), thickness=1):
    """在图像上叠加 GT 轮廓，返回 uint8 RGB."""
    vis = image_rgb.copy()
    for mask in gt_masks:
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, color, thickness)
    return vis


def boundary_heatmap(b_comp, colormap=cv2.COLORMAP_INFERNO):
    """将 [0,1] 浮点图转为伪彩色 uint8 RGB."""
    b_uint8 = np.clip(b_comp * 255.0, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(b_uint8, colormap)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def wunc_overlay(image_rgb, gt_masks, w_unc_arr, alpha=0.55):
    """将 W_unc 值叠加为 per-instance 填充色（蓝→红）在原图上."""
    overlay = image_rgb.copy().astype(np.float32)
    cmap = plt.get_cmap('RdYlGn')   # 绿=高置信, 红=低置信
    for i, (mask, w) in enumerate(zip(gt_masks, w_unc_arr)):
        color = np.array(cmap(float(w))[:3]) * 255.0   # RGB [0,255]
        overlay[mask] = alpha * color + (1 - alpha) * overlay[mask]
    return overlay.clip(0, 255).astype(np.uint8)


def make_diff_map(b_large, b_small):
    """绝对差异图，用 matplotlib coolwarm 伪彩色."""
    diff = b_large.astype(np.float32) - b_small.astype(np.float32)
    # 归一化到 [0,1] for colormap
    diff_norm = (diff + 1.0) / 2.0
    cmap = plt.get_cmap('coolwarm')
    rgba = cmap(diff_norm)   # (H, W, 4) float [0,1]
    return (rgba[:, :, :3] * 255).astype(np.uint8)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', default='data/gravel_big')
    p.add_argument('--split', default='train')
    p.add_argument('--num-images', type=int, default=25,
                   help='随机抽取的图片数量')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-inst-per-img', type=int, default=30,
                   help='每图最多处理的实例数（避免太慢）')
    p.add_argument('--out-dir', default='outputs/sam2_vis')
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data_root)
    split_dir = data_root / args.split
    ann_file = split_dir / '_annotations.coco.json'

    print(f"[INFO] Loading {ann_file}")
    with open(ann_file) as f:
        coco = json.load(f)

    ann_by_img = {}
    for ann in coco['annotations']:
        ann_by_img.setdefault(ann['image_id'], []).append(ann)

    images = coco['images']
    # 随机采样
    idxs = rng.choice(len(images), size=min(args.num_images, len(images)),
                      replace=False)
    selected = [images[i] for i in sorted(idxs)]
    print(f"[INFO] Selected {len(selected)} images")

    # ── 加载两个模型 ──────────────────────────────────────────────────────
    print(f"[INFO] Building SAM2-Large on {DEVICE}...")
    sam_large = build_sam2(SAM2_LARGE_CFG, SAM2_LARGE_CKPT, device=DEVICE)
    pred_large = SAM2ImagePredictor(sam_large)

    print(f"[INFO] Building SAM2-Small on {DEVICE}...")
    sam_small = build_sam2(SAM2_SMALL_CFG, SAM2_SMALL_CKPT, device=DEVICE)
    pred_small = SAM2ImagePredictor(sam_small)

    # 统计指标：每图 B_tea 均值，W_unc 均值
    stats_large, stats_small = [], []

    for img_info in tqdm(selected, desc='Processing'):
        img_path = split_dir / img_info['file_name']
        if not img_path.exists():
            # 尝试带路径前缀的情况
            img_path = data_root / img_info['file_name']
        if not img_path.exists():
            print(f"[WARN] Image not found: {img_path}, skip")
            continue

        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"[WARN] Cannot read {img_path}, skip")
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        H, W = image_rgb.shape[:2]

        anns = ann_by_img.get(img_info['id'], [])
        # 限制每图实例数，取面积最大的前 max_inst_per_img 个
        anns_valid = [a for a in anns
                      if a.get('segmentation') and not a.get('iscrowd', 0)]
        anns_valid.sort(key=lambda a: a.get('area', 0), reverse=True)
        anns_valid = anns_valid[:args.max_inst_per_img]

        if len(anns_valid) == 0:
            print(f"[WARN] No valid annotations for {img_info['file_name']}, skip")
            continue

        rng_l = np.random.default_rng(args.seed + img_info['id'])
        rng_s = np.random.default_rng(args.seed + img_info['id'] + 9999)

        b_large, w_large, gt_masks = run_sam2_on_image(
            pred_large, image_rgb, anns_valid, rng_l)
        b_small, w_small, _ = run_sam2_on_image(
            pred_small, image_rgb, anns_valid, rng_s)

        stats_large.append((b_large.mean(), w_large.mean() if len(w_large) else 0))
        stats_small.append((b_small.mean(), w_small.mean() if len(w_small) else 0))

        # ── 绘图：5 面板 ──────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 5, figsize=(28, 5))
        fig.suptitle(
            f"{img_info['file_name']}  ({W}×{H}, {len(anns_valid)} inst)",
            fontsize=9, y=1.01)

        # A: 原图 + GT 轮廓
        ax = axes[0]
        vis_gt = draw_gt_contours(image_rgb, gt_masks)
        ax.imshow(vis_gt)
        ax.set_title(f'GT Contours\n({len(anns_valid)} instances)', fontsize=8)
        ax.axis('off')

        # B: Large B_tea
        ax = axes[1]
        ax.imshow(boundary_heatmap(b_large))
        ax.set_title(
            f'Large B_tea\n'
            f'mean={b_large.mean():.4f}  W_unc={w_large.mean():.4f}',
            fontsize=8)
        ax.axis('off')

        # C: Small B_tea
        ax = axes[2]
        ax.imshow(boundary_heatmap(b_small))
        ax.set_title(
            f'Small B_tea\n'
            f'mean={b_small.mean():.4f}  W_unc={w_small.mean():.4f}',
            fontsize=8)
        ax.axis('off')

        # D: W_unc overlay (Large)
        ax = axes[3]
        w_overlay = wunc_overlay(image_rgb, gt_masks, w_large)
        ax.imshow(w_overlay)
        ax.set_title(
            f'W_unc (Large)\ngreen=confident  red=uncertain',
            fontsize=8)
        # 颜色条说明
        patches = [
            mpatches.Patch(color='green', label='W=1.0 (stable)'),
            mpatches.Patch(color='red',   label='W≈0.5 (noisy)')]
        ax.legend(handles=patches, loc='lower left', fontsize=6,
                  framealpha=0.6)
        ax.axis('off')

        # E: Large - Small 差异图
        ax = axes[4]
        ax.imshow(make_diff_map(b_large, b_small))
        abs_diff = np.abs(b_large - b_small)
        ax.set_title(
            f'Large − Small (|Δ| mean={abs_diff.mean():.4f})\n'
            f'blue=L<S  red=L>S',
            fontsize=8)
        ax.axis('off')

        plt.tight_layout()
        stem = Path(img_info['file_name']).stem
        save_path = out_dir / f"{stem}_sam2_compare.png"
        fig.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close(fig)

    # ── 汇总统计图 ────────────────────────────────────────────────────────
    if stats_large:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f'SAM2-Large vs SAM2-Small  |  {len(stats_large)} images  '
            f'|  max {args.max_inst_per_img} inst/img  |  K={K_RUNS} runs',
            fontsize=11)

        b_means_l = [s[0] for s in stats_large]
        b_means_s = [s[0] for s in stats_small]
        w_means_l = [s[1] for s in stats_large]
        w_means_s = [s[1] for s in stats_small]

        x = np.arange(len(stats_large))

        ax = axes[0]
        ax.plot(x, b_means_l, 'o-', label='Large', color='royalblue', ms=4)
        ax.plot(x, b_means_s, 's--', label='Small', color='tomato', ms=4)
        ax.set_title('B_tea composite mean per image')
        ax.set_xlabel('Image index')
        ax.set_ylabel('B_tea mean')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.text(0.02, 0.95,
                f'Δmean = {np.mean(b_means_l)-np.mean(b_means_s):.5f}',
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round', fc='wheat', alpha=0.5))

        ax = axes[1]
        ax.plot(x, w_means_l, 'o-', label='Large', color='royalblue', ms=4)
        ax.plot(x, w_means_s, 's--', label='Small', color='tomato', ms=4)
        ax.set_title('W_unc mean per image\n(higher = SAM2 more confident)')
        ax.set_xlabel('Image index')
        ax.set_ylabel('W_unc mean')
        ax.set_ylim(0.0, 1.05)
        ax.legend()
        ax.grid(alpha=0.3)
        ax.text(0.02, 0.05,
                f'Large W_unc={np.mean(w_means_l):.4f}\n'
                f'Small W_unc={np.mean(w_means_s):.4f}',
                transform=ax.transAxes, fontsize=9, va='bottom',
                bbox=dict(boxstyle='round', fc='wheat', alpha=0.5))

        plt.tight_layout()
        summary_path = out_dir / 'sam2_large_vs_small_summary.png'
        fig.savefig(summary_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"\n[DONE] Summary plot saved → {summary_path}")

        # 打印文字汇总
        print("\n" + "="*60)
        print(f"{'Metric':<30} {'Large':>10} {'Small':>10}")
        print("-"*60)
        print(f"{'B_tea mean':<30} {np.mean(b_means_l):>10.5f} {np.mean(b_means_s):>10.5f}")
        print(f"{'W_unc mean':<30} {np.mean(w_means_l):>10.4f} {np.mean(w_means_s):>10.4f}")
        print(f"{'|B_tea Δ| mean':<30} {np.mean(np.abs(np.array(b_means_l)-np.array(b_means_s))):>10.5f}")
        print("="*60)
        print(f"\n[DONE] {len(stats_large)} per-image plots → {out_dir}/")


if __name__ == '__main__':
    main()

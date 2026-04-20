#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from scipy import ndimage


SPLIT_LAYOUT = {
    'train': ('annotations/instances_train.json', 'train'),
    'val': ('annotations/instances_val.json', 'val'),
    'test': ('annotations/instances_test.json', 'test'),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate inter-instance boundary maps and distance '
        'transform maps for gravel_big_mmdet.')
    parser.add_argument(
        '--data-root',
        default='data/gravel_big_mmdet',
        help='Dataset root containing annotations/ and train|val|test/.')
    parser.add_argument(
        '--label-dir',
        default='auxiliary_labels',
        help='Output subdirectory under data-root for generated labels.')
    parser.add_argument(
        '--vis-dir',
        default='work_dirs_gravel_big/auxiliary_map_vis',
        help='Directory for visualization previews.')
    parser.add_argument(
        '--boundary-dilation-radius',
        type=int,
        default=4,
        help='Expansion radius in pixels used to grow unique instance seeds '
        'before extracting their separating seam map.')
    parser.add_argument(
        '--boundary-band-radius',
        type=int,
        default=2,
        help='Extra dilation radius used to convert the thin core seam into '
        'a support band for tolerant supervision.')
    parser.add_argument(
        '--save-dt-npy',
        action='store_true',
        help='Also save float32 distance transform maps as .npy files.')
    parser.add_argument(
        '--splits',
        nargs='+',
        choices=tuple(SPLIT_LAYOUT.keys()),
        default=list(SPLIT_LAYOUT.keys()),
        help='Dataset splits to generate. Defaults to all splits.')
    parser.add_argument(
        '--skip-preview',
        action='store_true',
        help='Skip preview image generation to accelerate offline label export.')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing labels and visualizations.')
    return parser.parse_args()


def load_coco(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def segmentation_to_mask(segmentation: Any, height: int, width: int) -> np.ndarray:
    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rles)
    elif isinstance(segmentation, dict) and isinstance(segmentation.get('counts'), list):
        rle = mask_utils.frPyObjects(segmentation, height, width)
    else:
        rle = segmentation

    mask = mask_utils.decode(rle)
    if mask.ndim == 3:
        mask = np.any(mask, axis=2)
    return mask.astype(bool)


def generate_inter_instance_boundary(
    instance_masks: list[np.ndarray],
    expansion_radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not instance_masks:
        raise ValueError('instance_masks must not be empty.')

    mask_stack = np.stack(instance_masks, axis=0).astype(bool)
    occupancy = mask_stack.sum(axis=0)
    height, width = instance_masks[0].shape
    valid_instance_count = int(np.count_nonzero(mask_stack.reshape(mask_stack.shape[0], -1).any(axis=1)))
    instance_density = valid_instance_count * 10000.0 / (height * width)

    if instance_density >= 10.0:
        seam_neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1))
        local_expansion_radius = max(1, expansion_radius - 1)
        gap_threshold = 1.0
    else:
        seam_neighbors = (
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (-1, -1), (1, -1), (-1, 1),
        )
        local_expansion_radius = expansion_radius
        gap_threshold = 1.0

    # Overlapped annotation pixels are treated as ambiguous and excluded from
    # the seeds so they do not create solid boundary blobs.
    label_map = np.zeros((height, width), dtype=np.int32)
    for idx, mask in enumerate(mask_stack, start=1):
        if not np.any(mask):
            continue

        unique_region = mask & (occupancy == 1)
        if np.any(unique_region):
            label_map[unique_region] = idx
            continue

        # If an annotation is fully overlapped by others, keep it alive with a
        # compact interior seed so it can still participate in seam extraction.
        mask_uint8 = mask.astype(np.uint8)
        dist = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
        if float(dist.max()) > 0:
            seed_y, seed_x = np.argwhere(dist == dist.max())[0]
        else:
            seed_y, seed_x = np.argwhere(mask_uint8 > 0)[0]
        label_map[seed_y, seed_x] = idx

    if not np.any(label_map):
        empty = np.zeros((height, width), dtype=np.uint8)
        return empty, empty, empty

    distances, indices = ndimage.distance_transform_edt(
        label_map == 0, return_indices=True)
    nearest_labels = label_map[tuple(indices)]
    expanded_labels = np.where(
        distances <= local_expansion_radius, nearest_labels, 0).astype(np.int32)

    exact_seam = np.zeros((height, width), dtype=bool)
    center = expanded_labels[1:-1, 1:-1]
    for dy, dx in seam_neighbors:
        neighbor = expanded_labels[
            1 + dy:height - 1 + dy,
            1 + dx:width - 1 + dx,
        ]
        exact_seam[1:-1, 1:-1] |= (
            (center > 0) & (neighbor > 0) & (center != neighbor)
        )

    # Treat soil gaps and direct-contact seams differently: soil gaps can be
    # widened slightly for learnability, while overlap/contact seams stay thin
    # so small stones are not swallowed by the target.
    all_instances_union = occupancy > 0
    overlap_region = occupancy >= 2

    background_seam = exact_seam & (~all_instances_union)
    overlap_seam = exact_seam & overlap_region

    # Only widen seams in sufficiently wide soil gaps. Narrow gaps stay as a
    # hairline so dense small stones are not overwhelmed by the label.
    background_gap_distance = ndimage.distance_transform_edt(~all_instances_union)
    wide_background_gap = background_gap_distance > gap_threshold

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thick_background_seam = cv2.dilate(
        background_seam.astype(np.uint8), kernel, iterations=1).astype(bool)
    thick_background_seam &= ~all_instances_union
    thick_background_seam &= wide_background_gap

    boundary_core = (background_seam | overlap_seam).astype(np.uint8)
    boundary_band = (background_seam | thick_background_seam | overlap_seam).astype(np.uint8)
    combined = boundary_band.copy()
    return boundary_core * 255, boundary_band * 255, combined * 255


def generate_distance_transform_map(instance_masks: list[np.ndarray]) -> np.ndarray:
    if not instance_masks:
        raise ValueError('instance_masks must not be empty.')

    dist_map = np.zeros(instance_masks[0].shape, dtype=np.float32)
    for mask in instance_masks:
        mask_uint8 = mask.astype(np.uint8)
        if mask_uint8.sum() == 0:
            continue
        dist = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
        max_value = float(dist.max())
        if max_value > 0:
            dist = dist / max_value
        np.maximum(dist_map, dist.astype(np.float32), out=dist_map)

    return dist_map


def write_map_png(path: Path, array: np.ndarray, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    ensure_parent(path)
    cv2.imwrite(str(path), array)


def write_npy(path: Path, array: np.ndarray, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    ensure_parent(path)
    np.save(path, array)


def make_preview(
    image_path: Path,
    boundary_core: np.ndarray,
    boundary_band: np.ndarray,
    dt_map: np.ndarray,
    output_path: Path,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        return

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f'Failed to read image: {image_path}')

    boundary_overlay = image.copy()
    band_pixels = boundary_band > 0
    core_pixels = boundary_core > 0
    boundary_overlay[band_pixels] = (
        0.55 * boundary_overlay[band_pixels] + 0.45 * np.array([0, 200, 255])
    ).astype(np.uint8)
    boundary_overlay[core_pixels] = (
        0.20 * boundary_overlay[core_pixels] + 0.80 * np.array([0, 0, 255])
    ).astype(np.uint8)

    dt_uint8 = np.clip(dt_map * 255.0, 0, 255).astype(np.uint8)
    dt_heatmap = cv2.applyColorMap(dt_uint8, cv2.COLORMAP_TURBO)
    dt_heatmap[dt_uint8 == 0] = 0

    preview = np.concatenate([image, boundary_overlay, dt_heatmap], axis=1)
    panel_width = image.shape[1]
    cv2.putText(preview, 'image', (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(preview, 'boundary core+band', (panel_width + 16, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                cv2.LINE_AA)
    cv2.putText(preview, 'distance-transform', (panel_width * 2 + 16, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                cv2.LINE_AA)

    ensure_parent(output_path)
    cv2.imwrite(str(output_path), preview)


def process_split(
    split: str,
    ann_path: Path,
    image_root: Path,
    label_root: Path,
    vis_root: Path,
    expansion_radius: int,
    band_radius: int,
    save_dt_npy: bool,
    skip_preview: bool,
    overwrite: bool,
) -> dict[str, Any]:
    coco = load_coco(ann_path)
    ann_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco.get('annotations', []):
        ann_by_image[int(ann['image_id'])].append(ann)

    split_summary: dict[str, Any] = {
        'images': 0,
        'annotations': len(coco.get('annotations', [])),
        'boundary_core_dir': str((label_root / split / 'boundary_core').as_posix()),
        'boundary_band_dir': str((label_root / split / 'boundary_band').as_posix()),
        'boundary_dir': str((label_root / split / 'boundary').as_posix()),
        'distance_dir': str((label_root / split / 'distance_transform').as_posix()),
        'preview_dir': str((vis_root / split).as_posix()),
        'manifests': [],
    }

    manifest: list[dict[str, Any]] = []
    for image_info in coco.get('images', []):
        image_id = int(image_info['id'])
        height = int(image_info['height'])
        width = int(image_info['width'])
        image_name = Path(image_info['file_name']).name
        image_path = image_root / image_name
        anns = ann_by_image[image_id]
        if not anns:
            continue

        instance_masks = [
            segmentation_to_mask(ann['segmentation'], height, width)
            for ann in anns
        ]
        boundary_core, boundary_band, boundary_map = generate_inter_instance_boundary(
            instance_masks, expansion_radius)
        if band_radius > 0:
            kernel_size = band_radius * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            boundary_band = cv2.dilate(boundary_core, kernel, iterations=1)
            boundary_band = np.maximum(boundary_band, boundary_map)
        dt_map = generate_distance_transform_map(instance_masks)

        boundary_core_path = label_root / split / 'boundary_core' / f'{Path(image_name).stem}.png'
        boundary_band_path = label_root / split / 'boundary_band' / f'{Path(image_name).stem}.png'
        boundary_path = label_root / split / 'boundary' / f'{Path(image_name).stem}.png'
        dt_png_path = label_root / split / 'distance_transform' / f'{Path(image_name).stem}.png'
        preview_path = vis_root / split / image_name

        write_map_png(boundary_core_path, boundary_core, overwrite)
        write_map_png(boundary_band_path, boundary_band, overwrite)
        write_map_png(boundary_path, boundary_map, overwrite)
        write_map_png(
            dt_png_path,
            np.clip(dt_map * 255.0, 0, 255).astype(np.uint8),
            overwrite)

        dt_npy_path = None
        if save_dt_npy:
            dt_npy_path = label_root / split / 'distance_transform_npy' / f'{Path(image_name).stem}.npy'
            write_npy(dt_npy_path, dt_map.astype(np.float32), overwrite)

        if not skip_preview:
            make_preview(image_path, boundary_core, boundary_band, dt_map, preview_path, overwrite)

        manifest.append({
            'image_id': image_id,
            'image_file': str(image_path.as_posix()),
            'boundary_core_file': str(boundary_core_path.as_posix()),
            'boundary_band_file': str(boundary_band_path.as_posix()),
            'boundary_file': str(boundary_path.as_posix()),
            'distance_transform_png': str(dt_png_path.as_posix()),
            'distance_transform_npy': str(dt_npy_path.as_posix()) if dt_npy_path else None,
            'preview_file': str(preview_path.as_posix()),
            'instance_count': len(instance_masks),
        })
        split_summary['images'] += 1

    manifest_path = label_root / split / 'manifest.json'
    ensure_parent(manifest_path)
    with manifest_path.open('w', encoding='utf-8') as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    split_summary['manifests'].append(str(manifest_path.as_posix()))
    split_summary['mean_instances_per_image'] = (
        float(sum(item['instance_count'] for item in manifest) / max(len(manifest), 1))
    )
    return split_summary


def main() -> None:
    args = parse_args()
    if args.boundary_dilation_radius < 1:
        raise ValueError('--boundary-dilation-radius must be >= 1')

    data_root = Path(args.data_root)
    label_root = data_root / args.label_dir
    vis_root = Path(args.vis_dir)

    summary: dict[str, Any] = {
        'data_root': str(data_root.as_posix()),
        'label_root': str(label_root.as_posix()),
        'visualization_root': str(vis_root.as_posix()),
        'boundary_dilation_radius': args.boundary_dilation_radius,
        'boundary_band_radius': args.boundary_band_radius,
        'save_dt_npy': args.save_dt_npy,
        'skip_preview': args.skip_preview,
        'splits': {},
    }

    for split in args.splits:
        ann_relpath, image_subdir = SPLIT_LAYOUT[split]
        summary['splits'][split] = process_split(
            split=split,
            ann_path=data_root / ann_relpath,
            image_root=data_root / image_subdir,
            label_root=label_root,
            vis_root=vis_root,
            expansion_radius=args.boundary_dilation_radius,
            band_radius=args.boundary_band_radius,
            save_dt_npy=args.save_dt_npy,
            skip_preview=args.skip_preview,
            overwrite=args.overwrite,
        )

    summary_path = label_root / 'generation_summary.json'
    ensure_parent(summary_path)
    with summary_path.open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
    
    
### 生成接缝线图
"""
1. 宏观：全局密度自适应（判断是“拥挤区”还是“稀疏区”）
◦ 代码算出了 instance_density。
◦ 如果石头很密（>= 10.0）：网络自动降低膨胀半径（expansion_radius - 1），且只用 4 邻域找缝隙。这就保证了在密密麻麻的小碎石堆里，接缝线“细如发丝”，绝对不会淹没小石头。
◦ 如果石头较稀疏：保持原半径，用 8 邻域，确保线段连贯。
2. 中观：精确的 Voronoi 势力划分
◦ 利用距离变换，绝对公平地找出每两块石头势力范围碰撞的“精确 1 像素分界线（exact_seam）”。
3. 微观：智能泥土填充（最绝的一步！）
◦ 代码把缝隙分成了两类：overlap_seam（两块石头物理上已经贴贴了）和 background_seam（两块石头中间隔着泥土）。
◦ 核心制约：它测量了泥土的宽度（background_gap_distance）。只有当泥土缝隙足够宽（> gap_threshold）时，它才允许对缝隙进行 3x3 的加粗（thick_background_seam）。如果是很窄的缝隙，保持 1 像素！
◦ 这完美做到了：宽缝隙填满泥土（增加正样本可学性），窄缝隙保留骨架（保护小目标主体）！
"""

### 生成实例距离变换图
"""
第一步：基于单实例的内部距离变换（Inner Distance Transform） 你没有把所有石头的 Mask 合并成一张大二值图再去算距离，而是在一层 for mask in instance_masks: 循环里单独为每一块石头算距离。
物理意义：它计算的是单块石头内部，每一个像素距离这块石头边缘的最短距离。越靠近石头中心，值越大；越靠近边缘，值越小（接近 0）。

第二步：★ 神级操作：实例级极值归一化 (Instance-level Normalization) ★ 你用了 dist = dist / max_value！这绝对是神来之笔！
如果不归一化会怎样？ 假设有一块巨大的石头，中心距离边缘 50 像素（峰值 50）；旁边有一块极小的碎石，中心距离边缘只有 3 像素（峰值 3）。如果直接输出，小石头在全局的距离图里几乎微不可见（才 0.06）。网络在学习时，会彻底忽略小石头，DF-FPN 也会因此对小目标失效。
归一化的巨大威力：通过除以 max_value，无论这块石头是占了半个屏幕的巨石，还是只有 5x5 像素的砂砾，它们的中心点（Core）都会被强制拉升到绝对的 1.0！边缘全部是 0.0！
对 DF-FPN 的意义：这完美实现了尺度无关性（Scale-Invariance）。你的网络学到的不再是绝对的“大小”，而是纯粹的“实例拓扑结构（Instance Topology）”——“不管多大，我只关心这里是石头的核心，还是石头的边缘。”

第三步：无损融合 (np.maximum) 最后把所有单块石头的距离图用 np.maximum 融合成一张图。哪怕两块石头的 Mask 在标注时有一点点重叠，取最大值也能保证保留最强烈的“核心属性”。
"""



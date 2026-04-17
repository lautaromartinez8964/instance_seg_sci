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
        '--save-dt-npy',
        action='store_true',
        help='Also save float32 distance transform maps as .npy files.')
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
) -> np.ndarray:
    if not instance_masks:
        raise ValueError('instance_masks must not be empty.')

    mask_stack = np.stack(instance_masks, axis=0).astype(bool)
    occupancy = mask_stack.sum(axis=0)
    height, width = instance_masks[0].shape

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
        return np.zeros((height, width), dtype=np.uint8)

    distances, indices = ndimage.distance_transform_edt(
        label_map == 0, return_indices=True)
    nearest_labels = label_map[tuple(indices)]
    expanded_labels = np.where(
        distances <= expansion_radius, nearest_labels, 0).astype(np.int32)

    exact_seam = np.zeros((height, width), dtype=bool)
    center = expanded_labels[1:-1, 1:-1]
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1),
                   (1, 1), (-1, -1), (1, -1), (-1, 1)):
        neighbor = expanded_labels[
            1 + dy:height - 1 + dy,
            1 + dx:width - 1 + dx,
        ]
        exact_seam[1:-1, 1:-1] |= (
            (center > 0) & (neighbor > 0) & (center != neighbor)
        )

    # Keep the exact Voronoi seam and widen it slightly for learnability,
    # while never letting the target drift into object interiors.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thick_seam = cv2.dilate(
        exact_seam.astype(np.uint8), kernel, iterations=1).astype(bool)

    all_instances_union = occupancy > 0
    overlap_region = occupancy >= 2
    boundary_final = thick_seam & ((~all_instances_union) | overlap_region)
    return boundary_final.astype(np.uint8) * 255


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
    boundary_map: np.ndarray,
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
    boundary_pixels = boundary_map > 0
    boundary_overlay[boundary_pixels] = (
        0.35 * boundary_overlay[boundary_pixels] + 0.65 * np.array([0, 0, 255])
    ).astype(np.uint8)

    dt_uint8 = np.clip(dt_map * 255.0, 0, 255).astype(np.uint8)
    dt_heatmap = cv2.applyColorMap(dt_uint8, cv2.COLORMAP_TURBO)
    dt_heatmap[dt_uint8 == 0] = 0

    preview = np.concatenate([image, boundary_overlay, dt_heatmap], axis=1)
    panel_width = image.shape[1]
    cv2.putText(preview, 'image', (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(preview, 'boundary', (panel_width + 16, 32),
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
    save_dt_npy: bool,
    overwrite: bool,
) -> dict[str, Any]:
    coco = load_coco(ann_path)
    ann_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco.get('annotations', []):
        ann_by_image[int(ann['image_id'])].append(ann)

    split_summary: dict[str, Any] = {
        'images': 0,
        'annotations': len(coco.get('annotations', [])),
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
        boundary_map = generate_inter_instance_boundary(
            instance_masks, expansion_radius)
        dt_map = generate_distance_transform_map(instance_masks)

        boundary_path = label_root / split / 'boundary' / f'{Path(image_name).stem}.png'
        dt_png_path = label_root / split / 'distance_transform' / f'{Path(image_name).stem}.png'
        preview_path = vis_root / split / image_name

        write_map_png(boundary_path, boundary_map, overwrite)
        write_map_png(
            dt_png_path,
            np.clip(dt_map * 255.0, 0, 255).astype(np.uint8),
            overwrite)

        dt_npy_path = None
        if save_dt_npy:
            dt_npy_path = label_root / split / 'distance_transform_npy' / f'{Path(image_name).stem}.npy'
            write_npy(dt_npy_path, dt_map.astype(np.float32), overwrite)

        make_preview(image_path, boundary_map, dt_map, preview_path, overwrite)

        manifest.append({
            'image_id': image_id,
            'image_file': str(image_path.as_posix()),
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
        'save_dt_npy': args.save_dt_npy,
        'splits': {},
    }

    for split, (ann_relpath, image_subdir) in SPLIT_LAYOUT.items():
        summary['splits'][split] = process_split(
            split=split,
            ann_path=data_root / ann_relpath,
            image_root=data_root / image_subdir,
            label_root=label_root,
            vis_root=vis_root,
            expansion_radius=args.boundary_dilation_radius,
            save_dt_npy=args.save_dt_npy,
            overwrite=args.overwrite,
        )

    summary_path = label_root / 'generation_summary.json'
    ensure_parent(summary_path)
    with summary_path.open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
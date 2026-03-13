#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert raw iSAID dataset into patch-based dataset compatible with MMDetection.

Input (recommended):
    iSAID_raw/
    ├── train/images/
    ├── val/images/
    └── test/images/

Annotations:
    OpenDataLab___iSAID/raw/train/Annotations/iSAID_train.json
    OpenDataLab___iSAID/raw/val/Annotations/iSAID_val.json

Output:
    iSAID_patches/
    ├── train/
    │   ├── images/
    │   └── instancesonly_filtered_train.json
    ├── val/
    │   ├── images/
    │   └── instancesonly_filtered_val.json
    └── test/
        └���─ images/
"""

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.errors import TopologicalError


def parse_args():
    parser = argparse.ArgumentParser(description='Convert iSAID to patch dataset')
    parser.add_argument(
        '--raw-root',
        type=str,
        required=True,
        help='Path to cleaned iSAID_raw directory')
    parser.add_argument(
        '--ann-root',
        type=str,
        required=True,
        help='Path to original annotation root, e.g. OpenDataLab___iSAID/raw')
    parser.add_argument(
        '--out-root',
        type=str,
        required=True,
        help='Output path for iSAID_patches')
    parser.add_argument(
        '--patch-size',
        type=int,
        default=800,
        help='Patch size, default 800')
    parser.add_argument(
        '--overlap',
        type=int,
        default=200,
        help='Patch overlap, default 200')
    parser.add_argument(
        '--modes',
        nargs='+',
        default=['train', 'val', 'test'],
        choices=['train', 'val', 'test'],
        help='Modes to process')
    parser.add_argument(
        '--save-mask-patches',
        action='store_true',
        help='Whether to save instance/semantic mask patches for train/val')
    parser.add_argument(
        '--min-area',
        type=float,
        default=4.0,
        help='Minimum polygon area after clipping')
    parser.add_argument(
        '--max-images',
        type=int,
        default=-1,
        help='Only process first N images for smoke test, -1 means all')
    return parser.parse_args()


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f)


def build_image_id_to_anns(coco):
    mapping = defaultdict(list)
    for ann in coco['annotations']:
        mapping[ann['image_id']].append(ann)
    return mapping


def get_stride(patch_size, overlap):
    return patch_size - overlap


def generate_patch_coords(width, height, patch_size, overlap):
    stride = get_stride(patch_size, overlap)

    if width <= patch_size:
        xs = [0]
    else:
        xs = list(range(0, width - patch_size + 1, stride))
        if xs[-1] != width - patch_size:
            xs.append(width - patch_size)

    if height <= patch_size:
        ys = [0]
    else:
        ys = list(range(0, height - patch_size + 1, stride))
        if ys[-1] != height - patch_size:
            ys.append(height - patch_size)

    return [(x, y) for y in ys for x in xs]


def list_base_images(image_dir):
    """
    Return only raw RGB images like P0000.png,
    excluding *_instance_id_RGB.png and *_instance_color_RGB.png
    """
    image_dir = Path(image_dir)
    files = sorted(image_dir.glob('*.png'))
    results = []
    for f in files:
        name = f.name
        if name.endswith('_instance_id_RGB.png'):
            continue
        if name.endswith('_instance_color_RGB.png'):
            continue
        results.append(f)
    return results


def read_rgb(path):
    return np.array(Image.open(path).convert('RGB'))


def save_rgb(arr, path):
    Image.fromarray(arr).save(path)


def crop_patch(img, x0, y0, patch_size):
    h, w = img.shape[:2]
    x1 = min(x0 + patch_size, w)
    y1 = min(y0 + patch_size, h)
    patch = img[y0:y1, x0:x1]

    # pad to fixed patch size if needed
    if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
        if img.ndim == 3:
            canvas = np.zeros((patch_size, patch_size, img.shape[2]), dtype=img.dtype)
        else:
            canvas = np.zeros((patch_size, patch_size), dtype=img.dtype)
        canvas[:patch.shape[0], :patch.shape[1]] = patch
        patch = canvas
    return patch


def segmentation_to_polygons(segmentation):
    """
    COCO polygon segmentation -> list[Polygon]
    segmentation format: [[x1, y1, x2, y2, ...], [...]]
    """
    polygons = []
    if not isinstance(segmentation, list):
        return polygons

    for seg in segmentation:
        if not isinstance(seg, list):
            continue
        if len(seg) < 6:
            continue
        pts = [(seg[i], seg[i + 1]) for i in range(0, len(seg), 2)]
        poly = Polygon(pts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            continue
        if poly.area <= 0:
            continue
        polygons.append(poly)
    return polygons


def geometry_to_coco_segmentation(geom, x0, y0, patch_size):
    """
    Convert shapely Polygon/MultiPolygon in global coords
    to COCO polygon segmentation in patch-local coords.
    """
    segs = []

    if geom.is_empty:
        return segs

    if isinstance(geom, Polygon):
        geoms = [geom]
    elif isinstance(geom, MultiPolygon):
        geoms = list(geom.geoms)
    else:
        return segs

    for poly in geoms:
        if poly.is_empty:
            continue
        exterior = list(poly.exterior.coords)
        if len(exterior) < 4:
            continue

        seg = []
        for x, y in exterior[:-1]:  # skip duplicated last point
            lx = max(0.0, min(float(x - x0), float(patch_size)))
            ly = max(0.0, min(float(y - y0), float(patch_size)))
            seg.extend([lx, ly])

        if len(seg) >= 6:
            segs.append(seg)

    return segs


def compute_bbox_from_segmentation(segmentation):
    xs, ys = [], []
    for seg in segmentation:
        for i in range(0, len(seg), 2):
            xs.append(seg[i])
            ys.append(seg[i + 1])

    if len(xs) == 0 or len(ys) == 0:
        return None

    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)

    w = max(0.0, x_max - x_min)
    h = max(0.0, y_max - y_min)

    return [float(x_min), float(y_min), float(w), float(h)]


def clip_annotation_to_patch(ann, patch_rect, x0, y0, patch_size, min_area):
    polygons = segmentation_to_polygons(ann['segmentation'])
    if len(polygons) == 0:
        return None

    clipped_parts = []
    total_area = 0.0

    for poly in polygons:
        try:
            inter = poly.intersection(patch_rect)
        except TopologicalError:
            poly = poly.buffer(0)
            inter = poly.intersection(patch_rect)

        if inter.is_empty:
            continue

        if isinstance(inter, Polygon):
            parts = [inter]
        elif isinstance(inter, MultiPolygon):
            parts = list(inter.geoms)
        else:
            continue

        for p in parts:
            if p.is_empty:
                continue
            if not p.is_valid:
                p = p.buffer(0)
            if p.is_empty:
                continue
            if p.area < min_area:
                continue
            clipped_parts.append(p)
            total_area += p.area

    if len(clipped_parts) == 0:
        return None

    if len(clipped_parts) == 1:
        merged = clipped_parts[0]
    else:
        merged = MultiPolygon(clipped_parts)

    segs = geometry_to_coco_segmentation(merged, x0, y0, patch_size)
    if len(segs) == 0:
        return None

    bbox = compute_bbox_from_segmentation(segs)
    if bbox is None:
        return None
    if bbox[2] <= 1 or bbox[3] <= 1:
        return None

    new_ann = {
        'segmentation': segs,
        'category_id': ann['category_id'],
        'category_name': ann.get('category_name', ''),
        'iscrowd': ann.get('iscrowd', 0),
        'area': float(total_area),
        'bbox': bbox
    }
    return new_ann


def process_test_mode(raw_root, out_root, patch_size, overlap, max_images):
    image_dir = Path(raw_root) / 'test' / 'images'
    out_img_dir = Path(out_root) / 'test' / 'images'
    ensure_dir(out_img_dir)

    base_images = list_base_images(image_dir)
    if max_images > 0:
        base_images = base_images[:max_images]

    print(f'[test] num base images: {len(base_images)}')

    for idx, img_path in enumerate(base_images):
        img = read_rgb(img_path)
        h, w = img.shape[:2]
        coords = generate_patch_coords(w, h, patch_size, overlap)

        stem = img_path.stem
        for x0, y0 in coords:
            patch = crop_patch(img, x0, y0, patch_size)
            patch_name = f'{stem}_{x0}_{y0}_{patch_size}_{patch_size}.png'
            save_rgb(patch, out_img_dir / patch_name)

        if (idx + 1) % 10 == 0 or idx == len(base_images) - 1:
            print(f'[test] processed {idx + 1}/{len(base_images)}')


def process_train_val_mode(mode, raw_root, ann_root, out_root, patch_size, overlap,
                           save_mask_patches, min_area, max_images):
    assert mode in ['train', 'val']

    image_dir = Path(raw_root) / mode / 'images'
    out_img_dir = Path(out_root) / mode / 'images'
    ensure_dir(out_img_dir)

    ann_json_name = f'iSAID_{mode}.json'
    ann_path = Path(ann_root) / mode / 'Annotations' / ann_json_name
    coco = load_json(ann_path)

    image_id_to_anns = build_image_id_to_anns(coco)
    categories = coco['categories']

    # map file_name -> image info
    images = coco['images']
    if max_images > 0:
        images = images[:max_images]

    new_images = []
    new_annotations = []

    new_image_id = 0
    new_ann_id = 0

    print(f'[{mode}] num base images: {len(images)}')

    for idx, image_info in enumerate(images):
        file_name = image_info['file_name']
        stem = Path(file_name).stem

        img_path = image_dir / file_name
        if not img_path.exists():
            raise FileNotFoundError(f'Image not found: {img_path}')

        img = read_rgb(img_path)
        h, w = img.shape[:2]

        ins_mask = None
        sem_mask = None
        if save_mask_patches:
            ins_path = image_dir / image_info.get('ins_file_name', f'{stem}_instance_id_RGB.png')
            sem_path = image_dir / image_info.get('seg_file_name', f'{stem}_instance_color_RGB.png')
            if ins_path.exists():
                ins_mask = np.array(Image.open(ins_path))
            if sem_path.exists():
                sem_mask = np.array(Image.open(sem_path))

        anns = image_id_to_anns[image_info['id']]
        coords = generate_patch_coords(w, h, patch_size, overlap)

        for x0, y0 in coords:
            patch_rect = box(x0, y0, x0 + patch_size, y0 + patch_size)

            patch_img = crop_patch(img, x0, y0, patch_size)
            patch_name = f'{stem}_{x0}_{y0}_{patch_size}_{patch_size}.png'
            patch_img_path = out_img_dir / patch_name
            save_rgb(patch_img, patch_img_path)

            if save_mask_patches and ins_mask is not None:
                ins_patch = crop_patch(ins_mask, x0, y0, patch_size)
                ins_name = f'{stem}_{x0}_{y0}_{patch_size}_{patch_size}_instance_id_RGB.png'
                Image.fromarray(ins_patch).save(out_img_dir / ins_name)

            if save_mask_patches and sem_mask is not None:
                sem_patch = crop_patch(sem_mask, x0, y0, patch_size)
                sem_name = f'{stem}_{x0}_{y0}_{patch_size}_{patch_size}_instance_color_RGB.png'
                Image.fromarray(sem_patch).save(out_img_dir / sem_name)

            patch_image_info = {
                'id': new_image_id,
                'width': patch_size,
                'height': patch_size,
                'file_name': patch_name,
                'orig_file_name': file_name,
                'orig_image_id': image_info['id'],
                'patch_x': x0,
                'patch_y': y0
            }

            if save_mask_patches:
                patch_image_info['ins_file_name'] = f'{stem}_{x0}_{y0}_{patch_size}_{patch_size}_instance_id_RGB.png'
                patch_image_info['seg_file_name'] = f'{stem}_{x0}_{y0}_{patch_size}_{patch_size}_instance_color_RGB.png'

            patch_ann_count = 0
            for ann in anns:
                clipped = clip_annotation_to_patch(
                    ann=ann,
                    patch_rect=patch_rect,
                    x0=x0,
                    y0=y0,
                    patch_size=patch_size,
                    min_area=min_area
                )
                if clipped is None:
                    continue

                clipped['id'] = new_ann_id
                clipped['image_id'] = new_image_id
                new_annotations.append(clipped)
                new_ann_id += 1
                patch_ann_count += 1

            # keep all patches for now; if you later want only non-empty patches, change here
            new_images.append(patch_image_info)
            new_image_id += 1

        if (idx + 1) % 10 == 0 or idx == len(images) - 1:
            print(f'[{mode}] processed {idx + 1}/{len(images)}')

    out_json = {
        'images': new_images,
        'categories': categories,
        'annotations': new_annotations
    }

    out_json_name = f'instancesonly_filtered_{mode}.json'
    out_json_path = Path(out_root) / mode / out_json_name
    ensure_dir(Path(out_root) / mode)
    save_json(out_json, out_json_path)

    print(f'[{mode}] patch images: {len(new_images)}')
    print(f'[{mode}] patch annotations: {len(new_annotations)}')
    print(f'[{mode}] saved to: {out_json_path}')


def main():
    args = parse_args()

    raw_root = Path(args.raw_root)
    ann_root = Path(args.ann_root)
    out_root = Path(args.out_root)

    ensure_dir(out_root)

    if 'train' in args.modes:
        process_train_val_mode(
            mode='train',
            raw_root=raw_root,
            ann_root=ann_root,
            out_root=out_root,
            patch_size=args.patch_size,
            overlap=args.overlap,
            save_mask_patches=args.save_mask_patches,
            min_area=args.min_area,
            max_images=args.max_images
        )

    if 'val' in args.modes:
        process_train_val_mode(
            mode='val',
            raw_root=raw_root,
            ann_root=ann_root,
            out_root=out_root,
            patch_size=args.patch_size,
            overlap=args.overlap,
            save_mask_patches=args.save_mask_patches,
            min_area=args.min_area,
            max_images=args.max_images
        )

    if 'test' in args.modes:
        process_test_mode(
            raw_root=raw_root,
            out_root=out_root,
            patch_size=args.patch_size,
            overlap=args.overlap,
            max_images=args.max_images
        )

    print('Done.')


if __name__ == '__main__':
    main()
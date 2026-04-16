#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


SPLIT_LAYOUT = {
    'train': ('train', 'instances_train.json'),
    'val': ('valid', 'instances_val.json'),
    'test': ('test', 'instances_test.json'),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Filter or merge a Roboflow/MMDetection COCO dataset into one class.')
    parser.add_argument('--input-root', required=True, help='Source dataset root.')
    parser.add_argument('--output-root', required=True, help='Output dataset root.')
    parser.add_argument(
        '--class-name', default='gravel', help='Category name to keep.')
    parser.add_argument(
        '--merge-all-categories',
        action='store_true',
        help='Map all source categories into the target class instead of filtering by name.')
    parser.add_argument(
        '--copy-mode',
        choices=['copy', 'hardlink'],
        default='copy',
        help='Whether to copy or hardlink image files into the output root.')
    parser.add_argument(
        '--drop-empty-images',
        action='store_true',
        help='Drop images with no remaining annotations after filtering.')
    parser.add_argument(
        '--overwrite', action='store_true', help='Remove output root before writing.')
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def prepare_output_root(output_root: Path, overwrite: bool) -> None:
    if output_root.exists() and overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'val', 'test', 'annotations'):
        (output_root / split).mkdir(parents=True, exist_ok=True)


def copy_image(src: Path, dst: Path, copy_mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_mode == 'hardlink':
        if dst.exists():
            dst.unlink()
        dst.hardlink_to(src)
        return
    shutil.copy2(src, dst)


def filter_split(
    split_name: str,
    src_dir: Path,
    output_root: Path,
    class_name: str,
    copy_mode: str,
    drop_empty_images: bool,
    merge_all_categories: bool,
) -> dict[str, int]:
    ann_path = src_dir / '_annotations.coco.json'
    coco = load_json(ann_path)

    if merge_all_categories:
        filtered_annotations = list(coco.get('annotations', []))
    else:
        keep_category = None
        for category in coco.get('categories', []):
            if category.get('name') == class_name:
                keep_category = category
                break
        if keep_category is None:
            raise ValueError(f'Class {class_name!r} not found in {ann_path}')

        keep_source_id = int(keep_category['id'])
        filtered_annotations = [
            ann for ann in coco.get('annotations', [])
            if int(ann['category_id']) == keep_source_id
        ]
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in filtered_annotations:
        annotations_by_image.setdefault(int(ann['image_id']), []).append(ann)

    output_images = []
    output_annotations = []
    next_image_id = 1
    next_ann_id = 1
    kept_images = 0

    for image in coco.get('images', []):
        source_image_id = int(image['id'])
        image_annotations = annotations_by_image.get(source_image_id, [])
        if drop_empty_images and not image_annotations:
            continue

        src_image_path = src_dir / image['file_name']
        if not src_image_path.exists():
            src_image_path = src_dir / Path(image['file_name']).name
        if not src_image_path.exists():
            raise FileNotFoundError(f'Missing source image: {src_dir / image["file_name"]}')

        dst_image_path = output_root / split_name / Path(image['file_name']).name
        copy_image(src_image_path, dst_image_path, copy_mode)

        output_images.append({
            'id': next_image_id,
            'file_name': dst_image_path.name,
            'width': int(image['width']),
            'height': int(image['height']),
        })

        for ann in image_annotations:
            ann_copy = {
                key: value
                for key, value in ann.items()
                if key not in {'id', 'image_id', 'category_id'}
            }
            ann_copy['id'] = next_ann_id
            ann_copy['image_id'] = next_image_id
            ann_copy['category_id'] = 1
            output_annotations.append(ann_copy)
            next_ann_id += 1

        next_image_id += 1
        kept_images += 1

    payload = {
        'info': coco.get('info', {'description': f'{class_name} filtered dataset'}),
        'licenses': coco.get('licenses', []),
        'images': output_images,
        'annotations': output_annotations,
        'categories': [{'id': 1, 'name': class_name, 'supercategory': 'none'}],
    }
    save_json(output_root / 'annotations' / SPLIT_LAYOUT[split_name][1], payload)

    return {
        'images': kept_images,
        'annotations': len(output_annotations),
        'empty_images_kept': kept_images - len({ann['image_id'] for ann in output_annotations}),
    }


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    if not input_root.exists():
        raise FileNotFoundError(f'Input root not found: {input_root}')

    prepare_output_root(output_root, overwrite=args.overwrite)

    summary: dict[str, Any] = {
        'input_root': str(input_root),
        'output_root': str(output_root),
        'class_name': args.class_name,
        'drop_empty_images': args.drop_empty_images,
        'merge_all_categories': args.merge_all_categories,
        'splits': {},
    }

    for split_name, (source_dir_name, _) in SPLIT_LAYOUT.items():
        src_dir = input_root / source_dir_name
        summary['splits'][split_name] = filter_split(
            split_name=split_name,
            src_dir=src_dir,
            output_root=output_root,
            class_name=args.class_name,
            copy_mode=args.copy_mode,
            drop_empty_images=args.drop_empty_images,
            merge_all_categories=args.merge_all_categories,
        )

    save_json(output_root / 'filter_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
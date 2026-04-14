#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
GRAVEL_CATEGORY = [{'id': 1, 'name': 'gravel', 'supercategory': 'none'}]
SPLIT_ALIASES = {
    'train': ('train',),
    'val': ('val', 'valid'),
    'test': ('test',),
}


@dataclass
class SplitLayout:
    split: str
    image_dir: Path
    ann_file: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Normalize one or merge two COCO gravel datasets into one MMDetection-friendly dataset.')
    parser.add_argument(
        '--dataset1',
        default=r'E:\yan1\data\gravel_merge\dataset1',
        help='Dataset root containing train/val/test for the 320x320 dataset.')
    parser.add_argument(
        '--dataset2',
        default=None,
        help='Optional dataset root containing train for the second gravel dataset.')
    parser.add_argument(
        '--output',
        default=r'E:\yan1\data\gravel_merge\merged_dataset',
        help='Merged dataset output directory.')
    parser.add_argument(
        '--dataset2-prefix',
        default='d2_',
        help='Filename prefix applied to dataset2 train images when copying.')
    parser.add_argument(
        '--copy-mode',
        choices=['copy', 'hardlink'],
        default='copy',
        help='Use copy for maximum compatibility, or hardlink to save disk space.')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Remove the output directory before merging.')
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as handle:
        data = json.load(handle)
    for key in ('images', 'annotations'):
        if key not in data or not isinstance(data[key], list):
            raise ValueError(f'{path} is missing COCO key: {key}')
    return data


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def find_annotation_file(dataset_root: Path, split: str) -> Path:
    aliases = SPLIT_ALIASES.get(split, (split,))
    candidates: list[Path] = []
    patterns = [*(dataset_root / alias for alias in aliases), dataset_root / 'annotations', dataset_root]
    for base in patterns:
        if not base.exists():
            continue
        for path in base.glob('*.json'):
            stem = path.stem.lower()
            parent_name = path.parent.name.lower()
            if any(alias in stem or parent_name == alias for alias in aliases):
                candidates.append(path)

    unique_candidates = sorted({path.resolve() for path in candidates})
    if not unique_candidates:
        raise FileNotFoundError(
            f'No COCO json was found for split "{split}" under {dataset_root}')
    if len(unique_candidates) == 1:
        return unique_candidates[0]

    scored: list[tuple[int, Path]] = []
    for path in unique_candidates:
        stem = path.stem.lower()
        parent_name = path.parent.name.lower()
        score = 0
        if any(alias in stem for alias in aliases):
            score += 4
        if any(parent_name == alias for alias in aliases):
            score += 3
        if 'instance' in stem or 'coco' in stem:
            score += 2
        if path.parent.name.lower() == 'annotations':
            score += 1
        scored.append((score, path))
    scored.sort(key=lambda item: (-item[0], str(item[1])))
    if len(scored) >= 2 and scored[0][0] == scored[1][0]:
        raise RuntimeError(
            f'Ambiguous annotation files for split "{split}": {unique_candidates}')
    return scored[0][1]


def infer_image_dir(dataset_root: Path, split: str, ann_file: Path) -> Path:
    aliases = SPLIT_ALIASES.get(split, (split,))
    for alias in aliases:
        preferred = dataset_root / alias
        if preferred.exists() and preferred.is_dir():
            return preferred

    for alias in aliases:
        sibling = ann_file.parent / alias
        if sibling.exists() and sibling.is_dir():
            return sibling

    raise FileNotFoundError(f'Cannot infer image directory for split "{split}" in {dataset_root}')


def resolve_layout(dataset_root: Path, split: str) -> SplitLayout:
    ann_file = find_annotation_file(dataset_root, split)
    image_dir = infer_image_dir(dataset_root, split, ann_file)
    return SplitLayout(split=split, image_dir=image_dir, ann_file=ann_file)


def prepare_output_dir(output_root: Path, overwrite: bool) -> None:
    if output_root.exists() and overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'val', 'test'):
        (output_root / split).mkdir(parents=True, exist_ok=True)
    (output_root / 'annotations').mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path, copy_mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_mode == 'hardlink':
        if dst.exists():
            dst.unlink()
        dst.hardlink_to(src)
    else:
        shutil.copy2(src, dst)


def unique_target_relpath(relpath: Path, used_relpaths: set[str], prefix: str = '') -> Path:
    parent = relpath.parent
    stem = relpath.stem
    suffix = relpath.suffix
    if suffix.lower() not in IMAGE_SUFFIXES:
        suffix = ''.join(relpath.suffixes)
    candidate = parent / f'{prefix}{relpath.name}'
    if str(candidate).lower() not in used_relpaths:
        used_relpaths.add(str(candidate).lower())
        return candidate

    serial = 1
    while True:
        candidate = parent / f'{prefix}{stem}_{serial}{suffix}'
        if str(candidate).lower() not in used_relpaths:
            used_relpaths.add(str(candidate).lower())
            return candidate
        serial += 1


def build_merged_coco(info: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    payload = {
        'info': info[0] if info else {'description': 'Merged gravel dataset'},
        'licenses': [],
        'images': [],
        'annotations': [],
        'categories': GRAVEL_CATEGORY,
    }
    return payload


def append_split_data(
    coco_data: dict[str, Any],
    src_img_dir: Path,
    dst_img_dir: Path,
    dst_coco: dict[str, Any],
    next_image_id: int,
    next_ann_id: int,
    used_relpaths: set[str],
    copy_mode: str,
    filename_prefix: str = '',
) -> tuple[int, int]:
    anns_by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in coco_data['annotations']:
        anns_by_image.setdefault(int(ann['image_id']), []).append(ann)

    for image in sorted(coco_data['images'], key=lambda item: int(item['id'])):
        original_relpath = Path(image['file_name'])
        src_path = src_img_dir / original_relpath
        if not src_path.exists():
            alt_path = src_img_dir / original_relpath.name
            if alt_path.exists():
                src_path = alt_path
                original_relpath = Path(original_relpath.name)
            else:
                raise FileNotFoundError(f'Missing source image: {src_path}')

        dst_relpath = unique_target_relpath(original_relpath, used_relpaths, prefix=filename_prefix)
        dst_path = dst_img_dir / dst_relpath
        copy_file(src_path, dst_path, copy_mode)

        image_id = next_image_id
        next_image_id += 1
        dst_coco['images'].append({
            'id': image_id,
            'file_name': dst_relpath.as_posix(),
            'width': int(image['width']),
            'height': int(image['height']),
        })

        for ann in sorted(anns_by_image.get(int(image['id']), []), key=lambda item: int(item['id'])):
            ann_copy = {key: value for key, value in ann.items() if key not in {'id', 'image_id', 'category_id'}}
            ann_copy['id'] = next_ann_id
            ann_copy['image_id'] = image_id
            ann_copy['category_id'] = 1
            dst_coco['annotations'].append(ann_copy)
            next_ann_id += 1

    return next_image_id, next_ann_id


def normalize_single_split(
    layout: SplitLayout,
    output_root: Path,
    copy_mode: str,
) -> tuple[int, int]:
    coco_data = load_json(layout.ann_file)
    dst_coco = build_merged_coco(info=[coco_data.get('info', {})])
    next_image_id, next_ann_id = append_split_data(
        coco_data=coco_data,
        src_img_dir=layout.image_dir,
        dst_img_dir=output_root / layout.split,
        dst_coco=dst_coco,
        next_image_id=1,
        next_ann_id=1,
        used_relpaths=set(),
        copy_mode=copy_mode,
    )
    save_json(output_root / 'annotations' / f'instances_{layout.split}.json', dst_coco)
    return next_image_id - 1, next_ann_id - 1


def merge_train_split(
    layout1: SplitLayout,
    layout2: SplitLayout,
    output_root: Path,
    copy_mode: str,
    dataset2_prefix: str,
) -> tuple[int, int]:
    coco1 = load_json(layout1.ann_file)
    coco2 = load_json(layout2.ann_file)
    dst_coco = build_merged_coco(info=[coco1.get('info', {}), coco2.get('info', {})])
    used_relpaths: set[str] = set()

    next_image_id = 1
    next_ann_id = 1
    next_image_id, next_ann_id = append_split_data(
        coco_data=coco1,
        src_img_dir=layout1.image_dir,
        dst_img_dir=output_root / 'train',
        dst_coco=dst_coco,
        next_image_id=next_image_id,
        next_ann_id=next_ann_id,
        used_relpaths=used_relpaths,
        copy_mode=copy_mode,
        filename_prefix='',
    )
    next_image_id, next_ann_id = append_split_data(
        coco_data=coco2,
        src_img_dir=layout2.image_dir,
        dst_img_dir=output_root / 'train',
        dst_coco=dst_coco,
        next_image_id=next_image_id,
        next_ann_id=next_ann_id,
        used_relpaths=used_relpaths,
        copy_mode=copy_mode,
        filename_prefix=dataset2_prefix,
    )
    save_json(output_root / 'annotations' / 'instances_train.json', dst_coco)
    return next_image_id - 1, next_ann_id - 1


def main() -> None:
    args = parse_args()
    dataset1 = Path(args.dataset1)
    dataset2 = Path(args.dataset2) if args.dataset2 else None
    output_root = Path(args.output)

    if not dataset1.exists():
        raise FileNotFoundError(f'dataset1 not found: {dataset1}')
    if dataset2 is not None and not dataset2.exists():
        raise FileNotFoundError(f'dataset2 not found: {dataset2}')

    prepare_output_dir(output_root, overwrite=args.overwrite)

    train1 = resolve_layout(dataset1, 'train')
    val1 = resolve_layout(dataset1, 'val')
    test1 = resolve_layout(dataset1, 'test')

    if dataset2 is None:
        train_images, train_anns = normalize_single_split(train1, output_root, args.copy_mode)
    else:
        train2 = resolve_layout(dataset2, 'train')
        train_images, train_anns = merge_train_split(
            layout1=train1,
            layout2=train2,
            output_root=output_root,
            copy_mode=args.copy_mode,
            dataset2_prefix=args.dataset2_prefix,
        )
    val_images, val_anns = normalize_single_split(val1, output_root, args.copy_mode)
    test_images, test_anns = normalize_single_split(test1, output_root, args.copy_mode)

    summary = {
        'output_root': str(output_root),
        'train': {'images': train_images, 'annotations': train_anns},
        'val': {'images': val_images, 'annotations': val_anns},
        'test': {'images': test_images, 'annotations': test_anns},
        'category': GRAVEL_CATEGORY[0],
    }
    save_json(output_root / 'merge_summary.json', summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
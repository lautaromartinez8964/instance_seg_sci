import argparse
import json
from pathlib import Path

import numpy as np
from mmengine.config import Config


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build a fixed gravel validation sample manifest by density buckets.')
    parser.add_argument('--config', help='Optional model config used to resolve a default output path.')
    parser.add_argument('--annotation', required=True, help='COCO annotation json.')
    parser.add_argument('--image-root', required=True, help='Validation image root.')
    parser.add_argument('--out', help='Output manifest json path. Defaults to <work_dir>/ibd_analysis/gravel_val_density_manifest.json.')
    parser.add_argument('--low-count', type=int, default=4, help='Number of low-density samples.')
    parser.add_argument('--mid-count', type=int, default=4, help='Number of medium-density samples.')
    parser.add_argument('--high-count', type=int, default=4, help='Number of high-density samples.')
    return parser.parse_args()


def resolve_manifest_path(config_path: str | None, out_path: str | None) -> Path:
    if out_path:
        return Path(out_path)
    if not config_path:
        raise ValueError('Either --out or --config must be provided.')

    cfg = Config.fromfile(config_path)
    work_dir = cfg.get('work_dir')
    if not work_dir:
        raise ValueError(f'Config does not define work_dir: {config_path}')
    return Path(work_dir) / 'ibd_analysis' / 'gravel_val_density_manifest.json'


def pick_evenly(records, num_items):
    if num_items <= 0 or not records:
        return []
    if num_items >= len(records):
        return list(records)

    positions = np.linspace(0, len(records) - 1, num=num_items, dtype=int)
    picks = []
    used = set()
    for position in positions.tolist():
        while position in used and position + 1 < len(records):
            position += 1
        used.add(position)
        picks.append(records[position])
    return picks


def main():
    args = parse_args()

    annotation_path = Path(args.annotation)
    image_root = Path(args.image_root)
    out_path = resolve_manifest_path(args.config, args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with annotation_path.open('r', encoding='utf-8') as handle:
        data = json.load(handle)

    ann_count = {}
    ann_area = {}
    for ann in data['annotations']:
        if ann.get('iscrowd', 0):
            continue
        image_id = ann['image_id']
        ann_count[image_id] = ann_count.get(image_id, 0) + 1
        ann_area[image_id] = ann_area.get(image_id, 0.0) + float(ann.get('area', 0.0))

    records = []
    for image_info in data['images']:
        image_id = image_info['id']
        width = int(image_info['width'])
        height = int(image_info['height'])
        area = max(width * height, 1)
        count = int(ann_count.get(image_id, 0))
        density_per_10k = count / area * 10000.0
        coverage_ratio = float(ann_area.get(image_id, 0.0)) / area
        records.append(
            dict(
                image_id=image_id,
                file_name=image_info['file_name'],
                image_path=str((image_root / Path(image_info['file_name']).name).as_posix()),
                width=width,
                height=height,
                instance_count=count,
                density_per_10k=density_per_10k,
                coverage_ratio=coverage_ratio,
            ))

    records.sort(key=lambda item: item['density_per_10k'])
    num_records = len(records)
    low_end = num_records // 3
    mid_end = (num_records * 2) // 3
    low_bucket = records[:low_end]
    mid_bucket = records[low_end:mid_end]
    high_bucket = records[mid_end:]

    selected = []
    bucket_specs = [
        ('low', low_bucket, args.low_count),
        ('mid', mid_bucket, args.mid_count),
        ('high', high_bucket, args.high_count),
    ]
    rank = 1
    for bucket_name, bucket_records, count in bucket_specs:
        for item in pick_evenly(bucket_records, count):
            sample = dict(item)
            sample['bucket'] = bucket_name
            sample['rank'] = rank
            rank += 1
            selected.append(sample)

    payload = dict(
        annotation=str(annotation_path.as_posix()),
        image_root=str(image_root.as_posix()),
        total_images=num_records,
        bucket_thresholds=dict(
            low_max_density=low_bucket[-1]['density_per_10k'] if low_bucket else None,
            mid_max_density=mid_bucket[-1]['density_per_10k'] if mid_bucket else None,
        ),
        samples=selected,
    )
    with out_path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    print(f'saved_manifest={out_path.as_posix()}')
    print(f'num_samples={len(selected)}')
    for bucket_name in ('low', 'mid', 'high'):
        bucket_samples = [item for item in selected if item['bucket'] == bucket_name]
        if not bucket_samples:
            continue
        densities = [item['density_per_10k'] for item in bucket_samples]
        print(
            f'{bucket_name}: count={len(bucket_samples)} '
            f'density_range=({min(densities):.4f}, {max(densities):.4f})')


if __name__ == '__main__':
    main()
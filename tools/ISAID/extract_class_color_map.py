import argparse
import json
from collections import defaultdict, Counter
from pathlib import Path

import cv2
import numpy as np
from pycocotools.coco import COCO


def mask_to_color_stats(color_img, mask):
    """
    Count RGB colors inside the given binary mask region.
    color_img: H x W x 3 RGB uint8 image
    mask: H x W binary mask
    """
    pixels = color_img[mask > 0]
    if len(pixels) == 0:
        return Counter()

    pixels = [tuple(map(int, p)) for p in pixels]
    return Counter(pixels)


def main():
    parser = argparse.ArgumentParser(
        description='Extract class-to-color mapping from iSAID *_instance_color_RGB.png maps')
    parser.add_argument(
        '--ann-file',
        required=True,
        help='Path to annotation json, e.g. data/iSAID/val/instancesonly_filtered_val.json')
    parser.add_argument(
        '--img-dir',
        required=True,
        help='Path to image directory, e.g. data/iSAID/val/images')
    parser.add_argument(
        '--out-file',
        default='outputs/isaid_class_color_map.json',
        help='Path to output mapping json')
    parser.add_argument(
        '--top-k-images',
        type=int,
        default=0,
        help='If > 0, only inspect the first top-k images for debugging')
    args = parser.parse_args()

    ann_file = Path(args.ann_file)
    img_dir = Path(args.img_dir)
    out_file = Path(args.out_file)

    if not ann_file.exists():
        raise FileNotFoundError(f'Annotation file not found: {ann_file}')
    if not img_dir.exists():
        raise FileNotFoundError(f'Image directory not found: {img_dir}')

    out_file.parent.mkdir(parents=True, exist_ok=True)

    coco = COCO(str(ann_file))

    catid_to_name = {
        c['id']: c['name']
        for c in coco.loadCats(coco.getCatIds())
    }

    img_ids = coco.getImgIds()
    if args.top_k_images > 0:
        img_ids = img_ids[:args.top_k_images]

    # Per-class observed colors
    class_color_counter = defaultdict(Counter)

    used_instances = 0
    skipped_instances = 0
    missing_color_map_files = 0

    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']

        # infer corresponding *_instance_color_RGB.png path
        if file_name.endswith('.png'):
            color_file_name = file_name.replace('.png', '_instance_color_RGB.png')
        else:
            color_file_name = file_name + '_instance_color_RGB.png'

        color_path = img_dir / color_file_name
        if not color_path.exists():
            missing_color_map_files += 1
            continue

        color_bgr = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
        if color_bgr is None:
            missing_color_map_files += 1
            continue

        color_img = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

        ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        anns = coco.loadAnns(ann_ids)

        for ann in anns:
            cat_id = ann['category_id']
            mask = coco.annToMask(ann)

            color_stats = mask_to_color_stats(color_img, mask)

            if len(color_stats) == 0:
                skipped_instances += 1
                continue

            # remove background if it sneaks in
            if (0, 0, 0) in color_stats:
                del color_stats[(0, 0, 0)]

            if len(color_stats) == 0:
                skipped_instances += 1
                continue

            dominant_color, _ = color_stats.most_common(1)[0]
            class_color_counter[cat_id][dominant_color] += 1
            used_instances += 1

    result = {}

    print('===== Extracted class-color mapping =====')
    for cat_id in sorted(class_color_counter.keys()):
        cat_name = catid_to_name.get(cat_id, f'class_{cat_id}')
        counter = class_color_counter[cat_id]
        dominant_color, count = counter.most_common(1)[0]

        result[str(cat_id)] = {
            'class_name': cat_name,
            'rgb': list(dominant_color),
            'support': int(count),
            'all_observed_colors': [
                {
                    'rgb': list(rgb),
                    'count': int(cnt)
                }
                for rgb, cnt in counter.most_common()
            ]
        }

        print(
            f'cat_id={cat_id:<2} '
            f'class={cat_name:<20} '
            f'rgb={dominant_color} '
            f'support={count}'
        )

    print()
    print(f'used_instances: {used_instances}')
    print(f'skipped_instances: {skipped_instances}')
    print(f'missing_color_map_files: {missing_color_map_files}')

    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f'\nSaved mapping to: {out_file}')


if __name__ == '__main__':
    main()
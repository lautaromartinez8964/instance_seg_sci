import argparse
import json
import math
import os
import re
from copy import deepcopy
from pathlib import Path

import mmcv
import numpy as np
from PIL import Image, ImageDraw
from pycocotools.coco import COCO

os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')

from mmengine.config import Config

from mmdet.apis import inference_detector, init_detector
from mmdet.registry import VISUALIZERS


PADDING = 8
LABEL_BAR_HEIGHT = 34


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare two instance segmentation models on sampled images.')
    parser.add_argument('config_a', help='Config file for model A.')
    parser.add_argument('checkpoint_a', help='Checkpoint file for model A.')
    parser.add_argument('config_b', help='Config file for model B.')
    parser.add_argument('checkpoint_b', help='Checkpoint file for model B.')
    parser.add_argument(
        '--annotation', required=True, help='COCO-style annotation json file.')
    parser.add_argument(
        '--image-root', required=True, help='Directory containing validation images.')
    parser.add_argument(
        '--out-dir', required=True, help='Output directory for comparison artifacts.')
    parser.add_argument('--label-a', default='model_a', help='Display label for model A.')
    parser.add_argument('--label-b', default='model_b', help='Display label for model B.')
    parser.add_argument(
        '--device', default='cuda:0', help='Inference device, e.g. cuda:0 or cpu.')
    parser.add_argument(
        '--pred-score-thr',
        type=float,
        default=0.3,
        help='Prediction score threshold for rendered instances.')
    parser.add_argument(
        '--num-images',
        type=int,
        default=20,
        help='Number of images to sample when no manifest is provided.')
    parser.add_argument(
        '--seed', type=int, default=20260408, help='Sampling seed.')
    parser.add_argument(
        '--sample-manifest',
        help='Optional existing manifest json. If set, the script reuses its sample list.')
    parser.add_argument(
        '--overview-cols',
        type=int,
        default=1,
        help='Number of columns in the top-level overview image.')
    return parser.parse_args()


def sanitize_label(label: str) -> str:
    slug = re.sub(r'[^0-9a-zA-Z]+', '_', label.strip()).strip('_').lower()
    return slug or 'model'


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def fallback_palette(num_colors: int):
    base = [
        (220, 20, 60),
        (0, 128, 255),
        (255, 165, 0),
        (60, 179, 113),
        (186, 85, 211),
        (255, 215, 0),
        (255, 99, 71),
        (70, 130, 180),
        (154, 205, 50),
        (255, 105, 180),
        (0, 206, 209),
        (199, 21, 133),
        (123, 104, 238),
        (255, 140, 0),
        (46, 139, 87),
    ]
    if num_colors <= len(base):
        return base[:num_colors]
    colors = list(base)
    for index in range(len(base), num_colors):
        colors.append(
            ((73 * index) % 255, (151 * index) % 255, (199 * index) % 255))
    return colors


def get_palette(dataset_meta: dict, num_classes: int):
    palette = dataset_meta.get('palette') if dataset_meta else None
    if palette is None:
        return fallback_palette(num_classes)
    return [tuple(color) for color in palette]


def build_visualizer(config_path: Path, dataset_meta: dict, name: str):
    cfg = Config.fromfile(str(config_path))
    vis_cfg = deepcopy(cfg.visualizer)
    vis_cfg['save_dir'] = None
    vis_cfg['name'] = name
    vis_cfg['vis_backends'] = []
    visualizer = VISUALIZERS.build(vis_cfg)
    visualizer.dataset_meta = dataset_meta
    return visualizer


def sample_entries(coco: COCO, image_root: Path, manifest_path: Path | None,
                   num_images: int, seed: int):
    images = coco.dataset['images']
    if manifest_path is not None:
        with open(manifest_path, 'r', encoding='utf-8') as handle:
            manifest = json.load(handle)
        entries = manifest['samples']
        resolved = []
        for entry in entries:
            img_path = Path(entry['img_path'])
            if not img_path.is_absolute():
                candidate = Path.cwd() / img_path
                if candidate.exists():
                    img_path = candidate
                else:
                    img_path = image_root / img_path.name
            if not img_path.exists():
                raise FileNotFoundError(f'Image from manifest does not exist: {entry["img_path"]}')
            resolved.append(
                dict(
                    rank=entry['rank'],
                    index=entry['index'],
                    img_id=entry['img_id'],
                    img_path=str(img_path),
                ))
        return resolved, manifest.get('seed', seed)

    total = len(images)
    sample_size = min(num_images, total)
    rng = np.random.default_rng(seed)
    indices = rng.choice(total, size=sample_size, replace=False).tolist()
    entries = []
    for rank, index in enumerate(indices, start=1):
        image_info = images[index]
        entries.append(
            dict(
                rank=rank,
                index=index,
                img_id=image_info['id'],
                img_path=str(image_root / Path(image_info['file_name']).name),
            ))
    return entries, seed


def build_category_color_index(coco: COCO, dataset_meta: dict):
    classes = list(dataset_meta.get('classes', [])) if dataset_meta else []
    class_to_index = {name: idx for idx, name in enumerate(classes)}
    color_index = {}
    for category in coco.dataset['categories']:
        category_name = category['name']
        if category_name in class_to_index:
            color_index[category['id']] = class_to_index[category_name]
        else:
            color_index[category['id']] = max(category['id'] - 1, 0)
    return color_index


def render_gt_overlay(image_path: Path, anns, coco: COCO, palette, color_index):
    base = Image.open(image_path).convert('RGB')
    canvas = np.array(base).astype(np.float32)
    for ann in anns:
        mask = coco.annToMask(ann).astype(bool)
        if not mask.any():
            continue
        color = np.array(palette[color_index[ann['category_id']] % len(palette)], dtype=np.float32)
        canvas[mask] = canvas[mask] * 0.45 + color * 0.55
    return Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))


def add_caption(image: Image.Image, text: str) -> Image.Image:
    canvas = Image.new('RGB', (image.width, image.height + LABEL_BAR_HEIGHT), (255, 255, 255))
    canvas.paste(image, (0, LABEL_BAR_HEIGHT))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, image.width, LABEL_BAR_HEIGHT), fill=(245, 245, 245))
    draw.text((10, 8), text, fill=(24, 24, 24))
    return canvas


def concat_horizontal(images):
    widths = [image.width for image in images]
    heights = [image.height for image in images]
    canvas = Image.new(
        'RGB',
        (sum(widths) + PADDING * (len(images) - 1), max(heights)),
        (255, 255, 255))
    offset_x = 0
    for image in images:
        canvas.paste(image, (offset_x, 0))
        offset_x += image.width + PADDING
    return canvas


def compose_grid(images, num_cols: int):
    num_cols = max(1, num_cols)
    num_rows = math.ceil(len(images) / num_cols)
    max_width = max(image.width for image in images)
    max_height = max(image.height for image in images)
    canvas = Image.new(
        'RGB',
        (num_cols * max_width + PADDING * (num_cols - 1),
         num_rows * max_height + PADDING * (num_rows - 1)),
        (255, 255, 255))
    for idx, image in enumerate(images):
        row = idx // num_cols
        col = idx % num_cols
        offset_x = col * (max_width + PADDING)
        offset_y = row * (max_height + PADDING)
        canvas.paste(image, (offset_x, offset_y))
    return canvas


def save_manifest(out_dir: Path, seed: int, entries, image_root: Path):
    payload = {'seed': seed, 'samples': []}
    for entry in entries:
        image_path = Path(entry['img_path'])
        try:
            manifest_path = str(image_path.resolve().relative_to(Path.cwd().resolve()))
        except ValueError:
            manifest_path = str(image_path)
        payload['samples'].append(
            dict(
                rank=entry['rank'],
                index=entry['index'],
                img_id=entry['img_id'],
                img_path=manifest_path,
            ))
    with open(out_dir / 'sample_manifest.json', 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)


def main():
    args = parse_args()

    annotation_path = Path(args.annotation)
    image_root = Path(args.image_root)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    label_a = args.label_a
    label_b = args.label_b
    file_label_a = sanitize_label(label_a)
    file_label_b = sanitize_label(label_b)

    coco = COCO(str(annotation_path))
    entries, manifest_seed = sample_entries(
        coco,
        image_root,
        Path(args.sample_manifest) if args.sample_manifest else None,
        args.num_images,
        args.seed,
    )

    model_a = init_detector(args.config_a, args.checkpoint_a, device=args.device)
    model_b = init_detector(args.config_b, args.checkpoint_b, device=args.device)

    visualizer_a = build_visualizer(
        Path(args.config_a), model_a.dataset_meta,
        f'compare_{file_label_a}_{os.getpid()}')
    visualizer_b = build_visualizer(
        Path(args.config_b), model_b.dataset_meta,
        f'compare_{file_label_b}_{os.getpid()}')

    num_classes = len(model_a.dataset_meta.get('classes', [])) if model_a.dataset_meta else 0
    palette = get_palette(model_a.dataset_meta, max(num_classes, 1))
    color_index = build_category_color_index(coco, model_a.dataset_meta)

    image_by_id = {image['id']: image for image in coco.dataset['images']}
    width = max(2, len(str(len(entries))))
    grid_paths = []

    for entry in entries:
        image_path = Path(entry['img_path'])
        image_info = image_by_id[entry['img_id']]
        sample_dir = out_dir / f"{entry['rank']:0{width}d}_{image_path.stem}"
        ensure_dir(sample_dir)

        input_image = Image.open(image_path).convert('RGB')
        input_image.save(sample_dir / 'input.png')

        anns = coco.loadAnns(coco.getAnnIds(imgIds=[entry['img_id']]))
        gt_image = render_gt_overlay(image_path, anns, coco, palette, color_index)
        gt_image.save(sample_dir / 'gt.png')

        result_a = inference_detector(model_a, str(image_path))
        result_b = inference_detector(model_b, str(image_path))

        visualizer_a.add_datasample(
            'compare_a',
            mmcv.imread(str(image_path)),
            data_sample=result_a,
            draw_gt=False,
            show=False,
            wait_time=0,
            pred_score_thr=args.pred_score_thr,
            out_file=str(sample_dir / f'{file_label_a}_pred.png'),
        )
        visualizer_b.add_datasample(
            'compare_b',
            mmcv.imread(str(image_path)),
            data_sample=result_b,
            draw_gt=False,
            show=False,
            wait_time=0,
            pred_score_thr=args.pred_score_thr,
            out_file=str(sample_dir / f'{file_label_b}_pred.png'),
        )

        compare_grid = concat_horizontal([
            add_caption(input_image, 'Input'),
            add_caption(gt_image, 'GT'),
            add_caption(Image.open(sample_dir / f'{file_label_a}_pred.png').convert('RGB'), label_a),
            add_caption(Image.open(sample_dir / f'{file_label_b}_pred.png').convert('RGB'), label_b),
        ])
        compare_grid_path = sample_dir / 'segmentation_compare_grid.png'
        compare_grid.save(compare_grid_path)
        grid_paths.append(compare_grid_path)

        with open(sample_dir / 'summary.json', 'w', encoding='utf-8') as handle:
            json.dump(
                dict(
                    rank=entry['rank'],
                    index=entry['index'],
                    img_id=entry['img_id'],
                    image=image_info['file_name'],
                    config_a=args.config_a,
                    checkpoint_a=args.checkpoint_a,
                    label_a=label_a,
                    config_b=args.config_b,
                    checkpoint_b=args.checkpoint_b,
                    label_b=label_b,
                ),
                handle,
                indent=2,
            )

    overview = compose_grid(
        [Image.open(path).convert('RGB') for path in grid_paths],
        args.overview_cols,
    )
    overview.save(out_dir / 'segmentation_compare_overview.png')
    save_manifest(out_dir, manifest_seed, entries, image_root)

    print(f'num_samples={len(entries)}')
    print(f'out_dir={out_dir}')
    print(f'overview={out_dir / "segmentation_compare_overview.png"}')
    print(f'manifest={out_dir / "sample_manifest.json"}')


if __name__ == '__main__':
    main()
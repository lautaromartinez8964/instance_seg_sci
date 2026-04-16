import argparse
import json
import math
import os
from copy import deepcopy
from pathlib import Path

os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')

import mmcv
import numpy as np
import torch
from PIL import Image, ImageDraw
from pycocotools.coco import COCO

from mmengine.config import Config

from mmdet.apis import inference_detector, init_detector
from mmdet.registry import VISUALIZERS
from mmdet.utils import register_all_modules


PADDING = 8
LABEL_BAR_HEIGHT = 34


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export prediction-vs-GT triptych visualizations.')
    parser.add_argument('config', help='Model config file.')
    parser.add_argument('checkpoint', help='Model checkpoint file.')
    parser.add_argument('--annotation', required=True, help='COCO annotation file.')
    parser.add_argument('--image-root', required=True, help='Image root directory.')
    parser.add_argument('--out-dir', required=True, help='Output directory.')
    parser.add_argument('--device', default='cuda:0', help='Inference device.')
    parser.add_argument(
        '--pred-score-thr',
        type=float,
        default=0.3,
        help='Prediction score threshold for rendered instances.')
    parser.add_argument(
        '--max-images',
        type=int,
        default=0,
        help='Optional image limit. Use 0 for all images.')
    parser.add_argument(
        '--overview-cols',
        type=int,
        default=1,
        help='Number of columns in the overview image.')
    return parser.parse_args()


def build_visualizer(config_path: Path, dataset_meta: dict, name: str):
    cfg = Config.fromfile(str(config_path))
    vis_cfg = deepcopy(cfg.visualizer)
    vis_cfg['save_dir'] = None
    vis_cfg['name'] = name
    vis_cfg['vis_backends'] = []
    visualizer = VISUALIZERS.build(vis_cfg)
    visualizer.dataset_meta = dataset_meta
    return visualizer


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
        color = np.array(
            palette[color_index[ann['category_id']] % len(palette)],
            dtype=np.float32)
        canvas[mask] = canvas[mask] * 0.45 + color * 0.55
    return Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))


def add_caption(image: Image.Image, text: str) -> Image.Image:
    canvas = Image.new(
        'RGB', (image.width, image.height + LABEL_BAR_HEIGHT), (255, 255, 255))
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


def patch_torch_load():
    orig_torch_load = torch.load

    def unsafe_load(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return orig_torch_load(*args, **kwargs)

    torch.load = unsafe_load


def main():
    args = parse_args()
    patch_torch_load()
    register_all_modules()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    coco = COCO(args.annotation)
    images = coco.dataset['images']
    if args.max_images > 0:
        images = images[:args.max_images]

    model = init_detector(args.config, args.checkpoint, device=args.device)
    visualizer = build_visualizer(Path(args.config), model.dataset_meta,
                                  f'gravel_triptych_{os.getpid()}')

    num_classes = len(model.dataset_meta.get('classes', [])) if model.dataset_meta else 0
    palette = get_palette(model.dataset_meta, max(num_classes, 1))
    color_index = build_category_color_index(coco, model.dataset_meta)

    rendered_paths = []
    manifest = []
    total = len(images)
    for index, image_info in enumerate(images, start=1):
        image_path = Path(args.image_root) / Path(image_info['file_name']).name
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[image_info['id']]))

        input_image = Image.open(image_path).convert('RGB')
        gt_image = render_gt_overlay(image_path, anns, coco, palette, color_index)

        pred_path = out_dir / 'pred_only' / f'{image_path.stem}.png'
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        result = inference_detector(model, str(image_path))
        visualizer.add_datasample(
            'prediction',
            mmcv.imread(str(image_path)),
            data_sample=result,
            draw_gt=False,
            show=False,
            wait_time=0,
            pred_score_thr=args.pred_score_thr,
            out_file=str(pred_path))

        pred_image = Image.open(pred_path).convert('RGB')
        triptych = concat_horizontal([
            add_caption(input_image, 'Input'),
            add_caption(gt_image, 'GT'),
            add_caption(pred_image, 'Prediction'),
        ])
        triptych_path = out_dir / 'triptychs' / f'{image_path.stem}.png'
        triptych_path.parent.mkdir(parents=True, exist_ok=True)
        triptych.save(triptych_path)
        rendered_paths.append(triptych_path)

        manifest.append(
            dict(
                index=index,
                total=total,
                img_id=image_info['id'],
                image_name=image_info['file_name'],
                triptych=str(triptych_path.relative_to(out_dir)),
                pred_only=str(pred_path.relative_to(out_dir)),
            ))

        print(f'[{index}/{total}] {image_info["file_name"]}')

    if rendered_paths:
        overview = compose_grid(
            [Image.open(path).convert('RGB') for path in rendered_paths],
            args.overview_cols)
        overview.save(out_dir / 'overview.png')

    with open(out_dir / 'manifest.json', 'w', encoding='utf-8') as handle:
        json.dump(
            dict(
                config=args.config,
                checkpoint=args.checkpoint,
                annotation=args.annotation,
                image_root=args.image_root,
                pred_score_thr=args.pred_score_thr,
                num_images=len(images),
                samples=manifest,
            ),
            handle,
            indent=2,
            ensure_ascii=False)

    print(f'out_dir={out_dir}')
    print(f'num_images={len(images)}')


if __name__ == '__main__':
    main()
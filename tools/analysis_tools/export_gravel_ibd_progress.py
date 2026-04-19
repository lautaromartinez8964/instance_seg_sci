import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.config import Config
from PIL import Image, ImageDraw

from mmdet.apis import inference_detector, init_detector
from mmdet.utils import register_all_modules


PADDING = 8
LABEL_BAR_HEIGHT = 34


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export IBD head visualization grids across checkpoints.')
    parser.add_argument('config', help='Model config file.')
    parser.add_argument('--checkpoint-dir', required=True, help='Directory containing epoch checkpoints.')
    parser.add_argument('--epochs', nargs='+', type=int, required=True, help='Epoch numbers to export.')
    parser.add_argument('--sample-manifest', required=True, help='Sample manifest json file.')
    parser.add_argument('--boundary-root', required=True, help='Boundary map root, e.g. auxiliary_labels/val/boundary.')
    parser.add_argument('--out-dir', help='Output directory. Defaults to <work_dir>/ibd_analysis/ibd_progress_epochs_...')
    parser.add_argument('--device', default='cuda:0', help='Inference device.')
    parser.add_argument('--binary-thr', type=float, default=0.5, help='Boundary probability threshold.')
    parser.add_argument('--overview-cols', type=int, default=2, help='Overview grid columns.')
    return parser.parse_args()


def resolve_out_dir(config_path: str, out_dir: str | None, epochs: list[int]) -> Path:
    if out_dir:
        return Path(out_dir)

    cfg = Config.fromfile(config_path)
    work_dir = cfg.get('work_dir')
    if not work_dir:
        raise ValueError(f'Config does not define work_dir: {config_path}')
    epoch_suffix = '_'.join(str(epoch) for epoch in epochs)
    return Path(work_dir) / 'ibd_analysis' / f'ibd_progress_epochs_{epoch_suffix}'


def patch_torch_load():
    original_load = torch.load

    def unsafe_load(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return original_load(*args, **kwargs)

    torch.load = unsafe_load


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_map(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32)
    return (array - array.min()) / max(array.max() - array.min(), 1e-6)


def colorize_heatmap(norm_map: np.ndarray) -> np.ndarray:
    norm_map = np.clip(norm_map, 0.0, 1.0)
    anchors = np.array([0.0, 0.33, 0.66, 1.0], dtype=np.float32)
    colors = np.array(
        [[20, 30, 120], [0, 190, 255], [255, 220, 0], [220, 30, 30]],
        dtype=np.float32)

    flat = norm_map.reshape(-1)
    rgb = np.zeros((flat.size, 3), dtype=np.float32)
    for index in range(len(anchors) - 1):
        left = anchors[index]
        right = anchors[index + 1]
        if index == len(anchors) - 2:
            mask = (flat >= left) & (flat <= right)
        else:
            mask = (flat >= left) & (flat < right)
        if not np.any(mask):
            continue
        weight = (flat[mask] - left) / max(right - left, 1e-6)
        rgb[mask] = colors[index] * (1.0 - weight[:, None]) + colors[index + 1] * weight[:, None]
    return rgb.reshape(norm_map.shape + (3,)).astype(np.uint8)


def blend_images(base: np.ndarray, overlay: np.ndarray, alpha: float) -> np.ndarray:
    return np.clip(base * (1.0 - alpha) + overlay * alpha, 0, 255).astype(np.uint8)


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


def get_backbone(model):
    backbone = model.backbone
    if hasattr(backbone, 'module'):
        return backbone.module
    return backbone


def load_boundary(boundary_path: Path) -> np.ndarray:
    boundary = mmcv.imread(str(boundary_path), flag='grayscale')
    if boundary is None:
        raise FileNotFoundError(f'Boundary map not found: {boundary_path.as_posix()}')
    return (boundary > 0).astype(np.uint8)


def resolve_image_path(sample: dict) -> Path:
    image_path = Path(sample['image_path'])
    if image_path.exists():
        return image_path
    candidate = Path.cwd() / image_path
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f'Image path not found for sample: {sample}')


def main():
    args = parse_args()
    patch_torch_load()
    register_all_modules()

    checkpoint_dir = Path(args.checkpoint_dir)
    boundary_root = Path(args.boundary_root)
    out_dir = resolve_out_dir(args.config, args.out_dir, args.epochs)
    ensure_dir(out_dir)

    with Path(args.sample_manifest).open('r', encoding='utf-8') as handle:
        manifest = json.load(handle)
    samples = manifest['samples']

    summary = dict(config=args.config, checkpoint_dir=str(checkpoint_dir), requested_epochs=args.epochs,
                   exported_epochs=[], missing_epochs=[], samples=[])
    for sample in samples:
        summary['samples'].append(
            dict(
                rank=sample['rank'],
                image_id=sample['image_id'],
                file_name=sample['file_name'],
                bucket=sample['bucket'],
                density_per_10k=sample['density_per_10k'],
                instance_count=sample['instance_count']))

    for epoch in args.epochs:
        checkpoint_path = checkpoint_dir / f'epoch_{epoch}.pth'
        if not checkpoint_path.exists():
            summary['missing_epochs'].append(dict(epoch=epoch, checkpoint=str(checkpoint_path)))
            continue

        model = init_detector(args.config, str(checkpoint_path), device=args.device)
        backbone = get_backbone(model)
        if not hasattr(backbone, 'get_last_boundary_logits'):
            raise RuntimeError('Backbone does not expose get_last_boundary_logits().')

        epoch_dir = out_dir / f'epoch_{epoch:02d}'
        ensure_dir(epoch_dir)
        grids = []
        epoch_stats = []

        for sample in samples:
            image_path = resolve_image_path(sample)
            boundary_path = boundary_root / f'{Path(sample["file_name"]).stem}.png'
            gt_boundary = load_boundary(boundary_path)

            image = Image.open(image_path).convert('RGB')
            image_np = np.array(image)

            with torch.no_grad():
                inference_detector(model, str(image_path))
            logits = backbone.get_last_boundary_logits()
            if logits is None:
                raise RuntimeError(f'IBD logits not captured for image: {image_path.as_posix()}')

            probability = torch.sigmoid(logits.float())
            probability = F.interpolate(
                probability,
                size=gt_boundary.shape,
                mode='bilinear',
                align_corners=False).squeeze().detach().cpu().numpy()
            probability = np.clip(probability, 0.0, 1.0)
            binary = (probability >= args.binary_thr).astype(np.uint8)

            heatmap = colorize_heatmap(probability)
            heat_overlay = blend_images(image_np.astype(np.float32), heatmap.astype(np.float32), alpha=0.42)
            binary_rgb = np.repeat(binary[:, :, None] * 255, 3, axis=2).astype(np.uint8)
            gt_rgb = np.repeat(gt_boundary[:, :, None] * 255, 3, axis=2).astype(np.uint8)

            panel = concat_horizontal([
                add_caption(image, 'Input'),
                add_caption(Image.fromarray(gt_rgb), 'GT boundary'),
                add_caption(Image.fromarray(heatmap), 'IBD heatmap'),
                add_caption(Image.fromarray(heat_overlay), 'IBD overlay'),
                add_caption(Image.fromarray(binary_rgb), f'IBD binary @{args.binary_thr:.2f}'),
            ])

            sample_dir = epoch_dir / f'{sample["rank"]:02d}_{Path(sample["file_name"]).stem}'
            ensure_dir(sample_dir)
            Image.fromarray(heatmap).save(sample_dir / 'ibd_heatmap.png')
            Image.fromarray(heat_overlay).save(sample_dir / 'ibd_overlay.png')
            Image.fromarray(binary_rgb).save(sample_dir / 'ibd_binary.png')
            Image.fromarray(gt_rgb).save(sample_dir / 'gt_boundary.png')
            image.save(sample_dir / 'input.png')
            panel_path = sample_dir / 'ibd_panel.png'
            panel.save(panel_path)
            grids.append(panel)

            gt_positive = int(gt_boundary.sum())
            pred_positive = int(binary.sum())
            intersection = int((binary & gt_boundary).sum())
            union = int(((binary | gt_boundary) > 0).sum())
            epoch_stats.append(
                dict(
                    rank=sample['rank'],
                    file_name=sample['file_name'],
                    bucket=sample['bucket'],
                    gt_positive=gt_positive,
                    pred_positive=pred_positive,
                    intersection=intersection,
                    union=union,
                    pred_mean=float(probability.mean()),
                    pred_max=float(probability.max()),
                ))

        if grids:
            compose_grid(grids, args.overview_cols).save(epoch_dir / 'overview.png')
        with (epoch_dir / 'stats.json').open('w', encoding='utf-8') as handle:
            json.dump(epoch_stats, handle, indent=2, ensure_ascii=False)

        summary['exported_epochs'].append(dict(epoch=epoch, checkpoint=str(checkpoint_path), out_dir=str(epoch_dir)))

    with (out_dir / 'summary.json').open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f'out_dir={out_dir.as_posix()}')
    print(f'exported_epochs={[item["epoch"] for item in summary["exported_epochs"]]}')
    if summary['missing_epochs']:
        print(f'missing_epochs={[item["epoch"] for item in summary["missing_epochs"]]}')


if __name__ == '__main__':
    main()
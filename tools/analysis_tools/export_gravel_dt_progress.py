import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')

import cv2
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
        description='Export DT-FPN guidance map visualization grids across checkpoints.')
    parser.add_argument('config', help='Model config file.')
    parser.add_argument('--checkpoint-dir', required=True, help='Directory containing epoch checkpoints.')
    parser.add_argument('--epochs', nargs='+', type=int, required=True, help='Epoch numbers to export.')
    parser.add_argument('--sample-manifest', required=True, help='Sample manifest json file.')
    parser.add_argument('--distance-root', required=True, help='Distance transform root, e.g. auxiliary_labels/val/distance_transform.')
    parser.add_argument('--out-dir', help='Output directory. Defaults to <work_dir>/dt_analysis/dt_progress_epochs_...')
    parser.add_argument('--device', default='cuda:0', help='Inference device.')
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
    return Path(work_dir) / 'dt_analysis' / f'dt_progress_epochs_{epoch_suffix}'


def patch_torch_load():
    original_load = torch.load

    def unsafe_load(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return original_load(*args, **kwargs)

    torch.load = unsafe_load


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def colorize_heatmap(norm_map: np.ndarray) -> np.ndarray:
    norm_map = np.clip(norm_map.astype(np.float32), 0.0, 1.0)
    heatmap = (norm_map * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


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


def get_neck(model):
    neck = model.neck
    if hasattr(neck, 'module'):
        return neck.module
    return neck


def resolve_level_name(neck, guided_level: int) -> str:
    start_level = getattr(neck, 'start_level', 0)
    return f'P{start_level + guided_level + 2}'


def collect_dt_predictions(neck) -> list[tuple[str, torch.Tensor]]:
    if hasattr(neck, 'get_last_dt_maps'):
        dt_maps = neck.get_last_dt_maps()
        if dt_maps:
            guided_levels = list(getattr(neck, 'guided_levels', range(len(dt_maps))))
            return [
                (resolve_level_name(neck, guided_level), dt_map)
                for guided_level, dt_map in zip(guided_levels, dt_maps)
            ]

    if hasattr(neck, 'get_last_dt_map'):
        dt_map = neck.get_last_dt_map()
        if dt_map is not None:
            return [('shared', dt_map)]

    return []


def load_distance(distance_path: Path) -> np.ndarray:
    distance = mmcv.imread(str(distance_path), flag='grayscale')
    if distance is None:
        raise FileNotFoundError(f'Distance transform map not found: {distance_path.as_posix()}')
    return distance.astype(np.float32) / 255.0


def resolve_image_path(sample: dict) -> Path:
    image_path = Path(sample['image_path'])
    if image_path.exists():
        return image_path
    candidate = Path.cwd() / image_path
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f'Image path not found for sample: {sample}')


def _normalize_scale_factor(scale_factor) -> tuple[float, float] | None:
    if scale_factor is None:
        return None
    if isinstance(scale_factor, (list, tuple)):
        if len(scale_factor) >= 2:
            return float(scale_factor[0]), float(scale_factor[1])
    return None


def crop_valid_dt_region(dt_map: torch.Tensor,
                         meta_info: dict | None,
                         gt_shape: tuple[int, int]) -> torch.Tensor:
    if meta_info is None:
        return dt_map

    ori_shape = meta_info.get('ori_shape')
    scale_factor = _normalize_scale_factor(meta_info.get('scale_factor'))
    if ori_shape is None or scale_factor is None:
        return dt_map

    reference_shape = meta_info.get('batch_input_shape') or meta_info.get(
        'pad_shape') or meta_info.get('img_shape')
    if reference_shape is None:
        return dt_map

    ori_h, ori_w = int(ori_shape[0]), int(ori_shape[1])
    scale_w, scale_h = scale_factor
    ref_h, ref_w = int(reference_shape[0]), int(reference_shape[1])
    resized_h = max(1, int(round(ori_h * scale_h)))
    resized_w = max(1, int(round(ori_w * scale_w)))
    valid_h = min(
        dt_map.shape[-2],
        max(1, int(round(dt_map.shape[-2] * resized_h / ref_h))))
    valid_w = min(
        dt_map.shape[-1],
        max(1, int(round(dt_map.shape[-1] * resized_w / ref_w))))

    # RTMDet uses keep_ratio + Pad to square inputs. The predicted DT map is
    # produced on the padded canvas, so we must remove the padded rows/cols
    # before projecting it back to the original image geometry.
    return dt_map[..., :valid_h, :valid_w]


def main():
    args = parse_args()
    patch_torch_load()
    register_all_modules()

    checkpoint_dir = Path(args.checkpoint_dir)
    distance_root = Path(args.distance_root)
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
        neck = get_neck(model)
        if not hasattr(neck, 'get_last_dt_map') and not hasattr(neck, 'get_last_dt_maps'):
            raise RuntimeError('Neck does not expose DT visualization interfaces.')

        epoch_dir = out_dir / f'epoch_{epoch:02d}'
        ensure_dir(epoch_dir)
        grids = []
        epoch_stats = []

        for sample in samples:
            image_path = resolve_image_path(sample)
            distance_path = distance_root / f'{Path(sample["file_name"]).stem}.png'
            gt_dt = load_distance(distance_path)

            image = Image.open(image_path).convert('RGB')
            image_np = np.array(image)

            with torch.no_grad():
                inference_result = inference_detector(model, str(image_path))
            dt_predictions = collect_dt_predictions(neck)
            if not dt_predictions:
                raise RuntimeError(f'DT map not captured for image: {image_path.as_posix()}')

            meta_info = getattr(inference_result, 'metainfo', None)

            gt_heatmap = colorize_heatmap(gt_dt)

            sample_dir = epoch_dir / f'{sample["rank"]:02d}_{Path(sample["file_name"]).stem}'
            ensure_dir(sample_dir)
            image.save(sample_dir / 'input.png')
            Image.fromarray(gt_heatmap).save(sample_dir / 'gt_dt_heatmap.png')

            panel_tiles = [
                add_caption(image, 'Input'),
                add_caption(Image.fromarray(gt_heatmap), 'GT DT'),
            ]
            level_stats = []
            for level_name, dt_map in dt_predictions:
                dt_map = crop_valid_dt_region(dt_map.float(), meta_info,
                                             gt_dt.shape)
                dt_map = F.interpolate(
                    dt_map,
                    size=gt_dt.shape,
                    mode='bilinear',
                    align_corners=False).squeeze().detach().cpu().numpy()
                dt_map = np.clip(dt_map, 0.0, 1.0)
                abs_error = np.abs(dt_map - gt_dt)

                pred_heatmap = colorize_heatmap(dt_map)
                err_heatmap = colorize_heatmap(abs_error)
                overlay = blend_images(
                    image_np.astype(np.float32),
                    pred_heatmap.astype(np.float32),
                    alpha=0.42)

                level_suffix = level_name.lower()
                Image.fromarray(pred_heatmap).save(
                    sample_dir / f'pred_dt_heatmap_{level_suffix}.png')
                Image.fromarray(overlay).save(
                    sample_dir / f'pred_dt_overlay_{level_suffix}.png')
                Image.fromarray(err_heatmap).save(
                    sample_dir / f'dt_abs_error_{level_suffix}.png')

                panel_tiles.extend([
                    add_caption(Image.fromarray(pred_heatmap), f'{level_name} Pred DT'),
                    add_caption(Image.fromarray(overlay), f'{level_name} overlay'),
                    add_caption(Image.fromarray(err_heatmap), f'{level_name} abs error'),
                ])
                level_stats.append(
                    dict(
                        level=level_name,
                        pred_mean=float(dt_map.mean()),
                        pred_max=float(dt_map.max()),
                        gt_mean=float(gt_dt.mean()),
                        gt_max=float(gt_dt.max()),
                        mae=float(abs_error.mean()),
                        mse=float(np.mean((dt_map - gt_dt) ** 2)),
                    ))

            panel = compose_grid(panel_tiles, 4)
            panel.save(sample_dir / 'dt_panel.png')
            grids.append(panel)

            epoch_stats.append(
                dict(
                    rank=sample['rank'],
                    file_name=sample['file_name'],
                    bucket=sample['bucket'],
                    levels=level_stats,
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
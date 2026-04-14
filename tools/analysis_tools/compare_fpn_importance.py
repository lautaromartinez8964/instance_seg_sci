import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')

from mmdet.apis import inference_detector, init_detector


DEFAULT_SAMPLE_NAMES = [
    'P0003_0_0_800_800',
    'P0003_0_223_800_800',
    'P0003_347_0_800_800',
    'P0003_347_223_800_800',
    'P0004_0_0_800_800',
    'P0004_0_600_800_800',
    'P0004_0_664_800_800',
    'P0004_24_0_800_800',
    'P0004_24_600_800_800',
    'P0004_24_664_800_800',
    'P0007_0_0_800_800',
    'P0007_0_1200_800_800',
    'P0007_0_1337_800_800',
    'P0007_0_600_800_800',
    'P0007_600_0_800_800',
    'P0007_600_1200_800_800',
    'P0007_600_1337_800_800',
    'P0007_600_600_800_800',
    'P0007_690_0_800_800',
    'P0007_690_1200_800_800',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize IG-FPN guidance maps with the same epoch/sample layout as previous IG-Scan visualizations.')
    parser.add_argument('config')
    parser.add_argument('checkpoint', nargs='?', default=None)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--grid-size', type=int, default=4)
    parser.add_argument('--val-image-dir', default='data/iSAID/val/images')
    parser.add_argument('--image', action='append', default=[])
    parser.add_argument('--image-list', default=None)
    parser.add_argument('--skip-existing', action='store_true')
    return parser.parse_args()


def normalize_map(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32)
    return (array - array.min()) / max(array.max() - array.min(), 1e-6)


def colorize_heatmap(norm_map: np.ndarray) -> np.ndarray:
    norm_map = np.clip(norm_map, 0.0, 1.0)
    anchors = np.array([0.0, 0.33, 0.66, 1.0], dtype=np.float32)
    colors = np.array(
        [
            [20, 30, 120],
            [0, 190, 255],
            [255, 220, 0],
            [220, 30, 30],
        ],
        dtype=np.float32,
    )

    flat = norm_map.reshape(-1)
    rgb = np.zeros((flat.size, 3), dtype=np.float32)
    for idx in range(len(anchors) - 1):
        left = anchors[idx]
        right = anchors[idx + 1]
        mask = (flat >= left) & (flat <= right if idx == len(anchors) - 2 else flat < right)
        if not np.any(mask):
            continue
        weight = (flat[mask] - left) / max(right - left, 1e-6)
        rgb[mask] = colors[idx] * (1.0 - weight[:, None]) + colors[idx + 1] * weight[:, None]
    return rgb.reshape(norm_map.shape + (3,)).astype(np.uint8)


def blend_images(base: np.ndarray, overlay: np.ndarray, alpha: float) -> np.ndarray:
    return np.clip(base * (1.0 - alpha) + overlay * alpha, 0, 255).astype(np.uint8)


def create_colorbar(height: int, width: int = 64) -> Image.Image:
    gradient = np.linspace(1.0, 0.0, height, dtype=np.float32)[:, None]
    heat = colorize_heatmap(np.repeat(gradient, width, axis=1))
    canvas = Image.fromarray(heat)
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), 'high', fill=(255, 255, 255))
    draw.text((8, height - 20), 'low', fill=(255, 255, 255))
    return canvas


def draw_grid_labels(canvas: Image.Image, rank_map: dict[int, int],
                     score_map: dict[int, float], gh: int, gw: int) -> None:
    draw = ImageDraw.Draw(canvas, 'RGBA')
    width, height = canvas.size
    cell_w = width / gw
    cell_h = height / gh
    for row in range(gh):
        for col in range(gw):
            left = col * cell_w
            top = row * cell_h
            right = (col + 1) * cell_w
            bottom = (row + 1) * cell_h
            region_id = row * gw + col
            rank = rank_map[region_id]
            score = score_map[region_id]
            draw.rectangle((left, top, right, bottom), outline=(255, 255, 0, 255), width=2)
            draw.rectangle((left + 2, top + 2, left + 72, top + 30), fill=(0, 0, 0, 140))
            draw.text((left + 6, top + 5), f'R{rank}', fill=(255, 255, 255, 255))
            draw.text((left + 34, top + 5), f'{score:.2f}', fill=(255, 230, 80, 255))


def resolve_images(args) -> list[Path]:
    resolved: list[Path] = []
    if args.image_list:
        with open(args.image_list, 'r', encoding='utf-8') as handle:
            for line in handle:
                item = line.strip()
                if item:
                    args.image.append(item)

    if args.image:
        raw_items = args.image
    else:
        raw_items = DEFAULT_SAMPLE_NAMES

    image_root = Path(args.val_image_dir)
    for item in raw_items:
        path = Path(item)
        if not path.suffix:
            path = image_root / f'{item}.png'
        elif not path.is_absolute():
            path = Path(item)
        resolved.append(path)
    return resolved


def build_pooled_state(guidance_map: torch.Tensor, grid_size: int):
    pooled = F.adaptive_avg_pool2d(guidance_map, (grid_size, grid_size)).flatten(1)
    order = torch.argsort(pooled, dim=-1, descending=True)
    return order, pooled, (grid_size, grid_size)


def extract_gate_stats(model) -> dict:
    stats = {}
    neck = getattr(model, 'neck', None)
    gates = getattr(neck, 'last_gates', ()) if neck is not None else ()
    level_names = ['p2', 'p3', 'p4', 'p5']
    for idx, gate in enumerate(gates):
        if gate is None:
            continue
        gate_cpu = gate.detach().float().cpu()
        name = level_names[idx] if idx < len(level_names) else f'gate_{idx}'
        stats[name] = {
            'shape': list(gate_cpu.shape),
            'min': float(gate_cpu.min()),
            'max': float(gate_cpu.max()),
            'mean': float(gate_cpu.mean()),
        }
    alpha = getattr(neck, 'alpha', None)
    if alpha is not None:
        stats['alpha'] = [float(x) for x in alpha.detach().float().cpu().tolist()]
    return stats


def save_guidance_overlay(image_path: Path, guidance_map: torch.Tensor,
                          order: torch.Tensor, pooled_scores: torch.Tensor,
                          gh: int, gw: int, out_dir: Path,
                          gate_stats: dict):
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image).astype(np.float32)

    guidance = guidance_map.squeeze().detach().cpu().numpy()
    guidance_norm = normalize_map(guidance)
    heat = Image.fromarray(colorize_heatmap(guidance_norm)).resize(image.size, Image.BILINEAR)
    heat_np = np.array(heat).astype(np.float32)
    Image.fromarray(heat_np.astype(np.uint8)).save(out_dir / 'importance_heatmap.png')

    overlay = Image.fromarray(blend_images(image_np, heat_np, alpha=0.42))

    order_list = order.squeeze().detach().cpu().tolist()
    score_list = pooled_scores.squeeze().detach().cpu().tolist()
    rank_map = {region_id: rank for rank, region_id in enumerate(order_list)}
    score_map = {region_id: float(score_list[region_id]) for region_id in range(len(score_list))}
    draw_grid_labels(overlay, rank_map, score_map, gh, gw)
    overlay.save(out_dir / 'importance_overlay.png')

    score_grid = np.array(score_list, dtype=np.float32).reshape(gh, gw)
    score_grid_norm = normalize_map(score_grid)
    score_canvas = Image.fromarray(colorize_heatmap(score_grid_norm)).resize(image.size, Image.NEAREST)
    score_overlay = Image.fromarray(
        blend_images(image_np, np.array(score_canvas).astype(np.float32), alpha=0.36))
    draw_grid_labels(score_overlay, rank_map, score_map, gh, gw)
    score_overlay.save(out_dir / 'region_score_overlay.png')

    colorbar = create_colorbar(image.size[1])
    panel = Image.new('RGB', (image.size[0] + colorbar.size[0], image.size[1]))
    panel.paste(overlay, (0, 0))
    panel.paste(colorbar, (image.size[0], 0))
    panel.save(out_dir / 'importance_overlay_with_colorbar.png')

    with open(out_dir / 'region_order.json', 'w', encoding='utf-8') as handle:
        json.dump(
            {
                'order': order_list,
                'grid': [gh, gw],
                'region_scores': score_list,
                'guidance_min': float(guidance.min()),
                'guidance_max': float(guidance.max()),
                'guidance_mean': float(guidance.mean()),
                'gate_stats': gate_stats,
            },
            handle,
            indent=2,
        )


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = init_detector(args.config, args.checkpoint, device=args.device)
    image_paths = resolve_images(args)

    for image_path in image_paths:
        if not image_path.exists():
            raise FileNotFoundError(f'Image not found: {image_path}')

        sample_out_dir = out_dir / image_path.stem
        if args.skip_existing and (sample_out_dir / 'importance_overlay.png').exists():
            print(f'[skip] {image_path.stem}')
            continue
        sample_out_dir.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            inference_detector(model, str(image_path))

        if not hasattr(model.backbone, 'get_last_guidance_map'):
            raise RuntimeError('Current backbone does not expose get_last_guidance_map().')
        guidance_map = model.backbone.get_last_guidance_map()
        if guidance_map is None:
            raise RuntimeError(
                f'Guidance map was not captured for {image_path.name}. '
                'Make sure the config uses output_guidance_map=True.')

        order, pooled_scores, grid_shape = build_pooled_state(guidance_map, args.grid_size)
        gate_stats = extract_gate_stats(model)
        save_guidance_overlay(
            image_path,
            guidance_map,
            order,
            pooled_scores,
            grid_shape[0],
            grid_shape[1],
            sample_out_dir,
            gate_stats,
        )
        print(f'[done] {image_path.stem}')


if __name__ == '__main__':
    main()
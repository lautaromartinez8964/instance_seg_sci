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
from mmdet.models.backbones.rs_lightmamba.ig_ss2d import IGSS2D


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize IG-Scan v2 states')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('image')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--device', default='cuda:0')
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


def infer_grid_shape(height: int, width: int, region_size: int) -> tuple[int, int]:
    gh = max(height // region_size, 1)
    gw = max(width // region_size, 1)
    gh = min(gh, height)
    gw = min(gw, width)

    while gh > 1 and height % gh != 0:
        gh -= 1
    while gw > 1 and width % gw != 0:
        gw -= 1
    return gh, gw


def derive_region_state(module: IGSS2D, importance: torch.Tensor):
    grid_shape = module.ig_scan_module.last_grid_shape
    region_scores = module.ig_scan_module.last_region_scores
    order = module.ig_scan_module.last_order

    if grid_shape is None:
        _, _, height, width = importance.shape
        grid_shape = infer_grid_shape(height, width, module.ig_scan_module.region_size)

    if region_scores is None:
        gh, gw = grid_shape
        region_scores = F.adaptive_avg_pool2d(importance, (gh, gw)).flatten(1)

    if order is None:
        order = torch.argsort(region_scores, dim=-1, descending=True)

    return order, region_scores, grid_shape


def save_importance_overlay(image_path: Path, importance: torch.Tensor,
                            order: torch.Tensor, region_scores: torch.Tensor,
                            gh: int, gw: int,
                            out_dir: Path):
    """
    保存前景重要性热力图叠加可视化结果。
    
    该函数将重要性分数与原始图像叠加生成可视化图，并在图像上绘制区域网格边界
    和每个区域的排序编号，同时将区域排序信息保存为 JSON 文件。
    
    Args:
        image_path (Path): 原始图像的文件路径
        importance (torch.Tensor): 重要性分数张量，形状为 (1, gh, gw)
        order (torch.Tensor): 区域排列顺序索引，形状为 (1, gh * gw)
        gh (int): 高度方向的网格划分数量
        gw (int): 宽度方向的网格划分数量
        out_dir (Path): 输出目录路径，用于保存可视化结果和 JSON 文件
    
    Returns:
        None
    
    Note:
        输出文件包括：
        - importance_overlay.png: 叠加了热力图、网格边界和排序编号的可视化图像
        - region_order.json: 包含区域排序顺序和网格尺寸的 JSON 文件
    """
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image).astype(np.float32)
    imp = importance.squeeze().detach().cpu().numpy()
    imp_norm = normalize_map(imp)
    heat = Image.fromarray(colorize_heatmap(imp_norm)).resize(image.size, Image.BILINEAR)
    heat_np = np.array(heat).astype(np.float32)

    pure_heatmap = Image.fromarray(heat_np.astype(np.uint8))
    pure_heatmap.save(out_dir / 'importance_heatmap.png')

    overlay = Image.fromarray(blend_images(image_np, heat_np, alpha=0.42))

    order_list = order.squeeze().detach().cpu().tolist()
    score_list = region_scores.squeeze().detach().cpu().tolist()
    rank_map = {region_id: rank for rank, region_id in enumerate(order_list)}
    score_map = {region_id: float(score_list[region_id]) for region_id in range(len(score_list))}
    draw_grid_labels(overlay, rank_map, score_map, gh, gw)
    overlay.save(out_dir / 'importance_overlay.png')

    score_grid = np.array(score_list, dtype=np.float32).reshape(gh, gw)
    score_grid_norm = normalize_map(score_grid)
    score_canvas = Image.fromarray(colorize_heatmap(score_grid_norm)).resize(image.size, Image.NEAREST)
    score_overlay = Image.fromarray(blend_images(image_np, np.array(score_canvas).astype(np.float32), alpha=0.36))
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
                'importance_min': float(imp.min()),
                'importance_max': float(imp.max()),
            },
            handle,
            indent=2,
        )


def main():
    """
    主函数：执行模型推理并可视化 IG-Scan 模块的前景重要性图
    
    该函数初始化检测模型，定位其中的 IGSS2D 模块，对输入图像执行推理，
    捕获 IG-Scan 模块在推理过程中生成的重要性分数、区域排列顺序和网格形状，
    并将结果保存为可视化图像。
    
    Args:
        无直接参数，通过命令行参数解析获取：
            - config: 模型配置文件路径
            - checkpoint: 模型权重文件路径
            - device: 推理设备（如 'cuda:0' 或 'cpu'）
            - image: 输入图像路径
            - out_dir: 输出目录路径
    
    Raises:
        RuntimeError: 当模型中未找到 IGSS2D 模块时抛出
        RuntimeError: 当推理过程中未能捕获 IG-Scan 状态时抛出
    """
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = init_detector(args.config, args.checkpoint, device=args.device)
    target_module = None
    for module in model.modules():
        if isinstance(module, IGSS2D):
            target_module = module
    if target_module is None:
        raise RuntimeError('No IGSS2D module found in model.')

    with torch.no_grad():
        inference_detector(model, args.image)

    importance = target_module.ig_scan_module.last_importance
    if importance is None:
        raise RuntimeError('IG-Scan state was not captured during inference.')

    order, region_scores, grid_shape = derive_region_state(target_module, importance)

    gh, gw = grid_shape
    save_importance_overlay(
        Path(args.image),
        importance,
        order,
        region_scores,
        gh,
        gw,
        out_dir,
    )


if __name__ == '__main__':
    main()
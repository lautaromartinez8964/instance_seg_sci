import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from mmdet.apis import init_detector, inference_detector


def encode_instance_id_to_rgb(instance_id: int):
    """
    Encode instance id to RGB triplet.
    """
    r = instance_id & 255
    g = (instance_id >> 8) & 255
    b = (instance_id >> 16) & 255
    return np.array([r, g, b], dtype=np.uint8)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export prediction maps in iSAID-style formats')
    parser.add_argument('img', help='Path to one input image')
    parser.add_argument('config', help='MMDetection config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--class-color-map',
        required=True,
        help='Path to extracted class color map json')
    parser.add_argument(
        '--out-dir',
        default='outputs_mask_rcnn_resnet50/pred_isaid_maps',
        help='Output directory')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.3,
        help='Score threshold')
    parser.add_argument(
        '--device',
        default='cuda:0',
        help='Device for inference')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.5,
        help='Overlay alpha')
    return parser.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_path = Path(args.img)
    base_name = img_path.stem

    # Load color map
    with open(args.class_color_map, 'r', encoding='utf-8') as f:
        class_color_map = json.load(f)

    # Convert class_color_map keys to int -> np.uint8 RGB
    classid_to_rgb = {}
    for k, v in class_color_map.items():
        class_id = int(k)
        rgb = np.array(v['rgb'], dtype=np.uint8)
        classid_to_rgb[class_id] = rgb

    # Load model
    model = init_detector(args.config, args.checkpoint, device=args.device)

    # Read original image
    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f'Cannot read image: {img_path}')
    h, w = img_bgr.shape[:2]

    # Inference
    result = inference_detector(model, str(img_path))
    pred = result.pred_instances

    # Prepare blank outputs
    pred_color_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    pred_id_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    overlay_bgr = img_bgr.copy()

    if len(pred) == 0:
        print('No predicted instances found.')
    else:
        scores = pred.scores.detach().cpu().numpy()
        labels = pred.labels.detach().cpu().numpy()
        masks = pred.masks.detach().cpu().numpy().astype(bool)

        keep = scores >= args.score_thr
        scores = scores[keep]
        labels = labels[keep]
        masks = masks[keep]

        # Sort by score descending
        order = np.argsort(-scores)
        scores = scores[order]
        labels = labels[order]
        masks = masks[order]

        occupied = np.zeros((h, w), dtype=bool)

        instance_counter = 1
        for score, label, mask in zip(scores, labels, masks):
            # MMDet labels are 0-based, your extracted cat_ids are 1-based
            class_id = int(label) + 1

            # do not overwrite higher-score instances
            mask = mask & (~occupied)
            if not mask.any():
                continue

            # instance_color_RGB: class-based color
            rgb_color = classid_to_rgb.get(class_id, np.array([255, 255, 255], dtype=np.uint8))
            pred_color_rgb[mask] = rgb_color

            # instance_id_RGB: instance-unique RGB code
            rgb_id = encode_instance_id_to_rgb(instance_counter)
            pred_id_rgb[mask] = rgb_id

            # overlay (convert RGB color to BGR for cv2 image)
            bgr_color = rgb_color[::-1].astype(np.float32)
            overlay_bgr[mask] = (
                args.alpha * bgr_color
                + (1.0 - args.alpha) * overlay_bgr[mask].astype(np.float32)
            ).astype(np.uint8)

            occupied[mask] = True
            instance_counter += 1

    # Save outputs
    pred_color_bgr = cv2.cvtColor(pred_color_rgb, cv2.COLOR_RGB2BGR)
    pred_id_bgr = cv2.cvtColor(pred_id_rgb, cv2.COLOR_RGB2BGR)

    color_path = out_dir / f'{base_name}_pred_instance_color_RGB.png'
    id_path = out_dir / f'{base_name}_pred_instance_id_RGB.png'
    overlay_path = out_dir / f'{base_name}_pred_overlay.png'

    cv2.imwrite(str(color_path), pred_color_bgr)
    cv2.imwrite(str(id_path), pred_id_bgr)
    cv2.imwrite(str(overlay_path), overlay_bgr)

    print(f'Saved: {color_path}')
    print(f'Saved: {id_path}')
    print(f'Saved: {overlay_path}')


if __name__ == '__main__':
    main()
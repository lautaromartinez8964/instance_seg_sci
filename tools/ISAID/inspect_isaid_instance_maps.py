import argparse
from pathlib import Path

import cv2
import numpy as np


def rgb_to_int(rgb):
    """Pack RGB to a single int for uniqueness analysis."""
    rgb = rgb.astype(np.uint32)
    return (rgb[..., 0] << 16) + (rgb[..., 1] << 8) + rgb[..., 2]


def describe_image(name, img):
    print(f'===== {name} =====')
    print(f'shape: {img.shape}')
    print(f'dtype: {img.dtype}')
    print(f'min: {img.min()}, max: {img.max()}')

    flat = img.reshape(-1, img.shape[-1])
    uniq = np.unique(flat, axis=0)
    print(f'unique RGB colors: {len(uniq)}')

    # show first 20 unique colors
    print('first 20 unique RGB values:')
    for c in uniq[:20]:
        print(c.tolist())

    packed = rgb_to_int(img)
    uniq_packed = np.unique(packed)
    print(f'unique packed RGB values: {len(uniq_packed)}')
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('color_map', help='Path to *_instance_color_RGB.png')
    parser.add_argument('id_map', help='Path to *_instance_id_RGB.png')
    args = parser.parse_args()

    color_img = cv2.imread(args.color_map, cv2.IMREAD_COLOR)
    id_img = cv2.imread(args.id_map, cv2.IMREAD_COLOR)

    if color_img is None:
        raise FileNotFoundError(args.color_map)
    if id_img is None:
        raise FileNotFoundError(args.id_map)

    # cv2 reads BGR, convert to RGB for interpretation
    color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
    id_img = cv2.cvtColor(id_img, cv2.COLOR_BGR2RGB)

    describe_image('instance_color_RGB', color_img)
    describe_image('instance_id_RGB', id_img)

    same = np.array_equal(color_img, id_img)
    print(f'Are the two images exactly identical? {same}')

    # how many pixels differ?
    diff = np.any(color_img != id_img, axis=-1)
    print(f'Num different pixels: {diff.sum()} / {diff.size}')

    # background stats
    bg_color = np.array([0, 0, 0], dtype=np.uint8)
    color_bg = np.all(color_img == bg_color, axis=-1).sum()
    id_bg = np.all(id_img == bg_color, axis=-1).sum()
    print(f'Background pixels in color map: {color_bg}')
    print(f'Background pixels in id map: {id_bg}')


if __name__ == '__main__':
    main()
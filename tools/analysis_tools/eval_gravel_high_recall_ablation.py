import argparse
import json
import os
from pathlib import Path

import mmcv
import torch
from mmengine.config import Config
from mmengine.logging import MMLogger
from mmengine.utils import ProgressBar, mkdir_or_exist

from mmdet.apis import inference_detector, init_detector
from mmdet.evaluation.metrics.coco_metric import CocoMetric
from mmdet.utils.large_image import merge_results_by_nms

try:
    from mmengine.logging.history_buffer import HistoryBuffer
    torch.serialization.add_safe_globals([HistoryBuffer])
except ImportError:
    HistoryBuffer = None

torch.serialization.add_safe_globals(['mmengine.logging.history_buffer.HistoryBuffer'])

_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

try:
    from sahi.slicing import slice_image
except ImportError as exc:
    raise ImportError(
        'Please install sahi before running sliced evaluation.') from exc


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate high-recall inference ablations on gravel val.')
    parser.add_argument('config', help='Model config path.')
    parser.add_argument('checkpoint', help='Checkpoint path.')
    parser.add_argument(
        '--ann-file',
        default='data/gravel_big_mmdet/annotations/instances_val.json',
        help='COCO annotation file for evaluation.')
    parser.add_argument(
        '--img-root',
        default='data/gravel_big_mmdet/val',
        help='Image root directory matching ann-file.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference.')
    parser.add_argument(
        '--mode',
        choices=['full', 'slice'],
        default='full',
        help='Use full-image or sliced inference.')
    parser.add_argument(
        '--scale',
        type=int,
        default=640,
        help='Resize scale applied by the test pipeline.')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.05,
        help='RCNN score threshold.')
    parser.add_argument(
        '--max-per-img',
        type=int,
        default=100,
        help='RCNN max_per_img during test.')
    parser.add_argument(
        '--rcnn-nms-iou',
        type=float,
        default=0.5,
        help='RCNN NMS IoU threshold.')
    parser.add_argument(
        '--rpn-nms-pre',
        type=int,
        default=1000,
        help='RPN nms_pre during test.')
    parser.add_argument(
        '--rpn-max-per-img',
        type=int,
        default=1000,
        help='RPN max_per_img during test.')
    parser.add_argument(
        '--rpn-nms-iou',
        type=float,
        default=0.7,
        help='RPN NMS IoU threshold.')
    parser.add_argument(
        '--patch-size',
        type=int,
        default=320,
        help='Patch size used only in slice mode.')
    parser.add_argument(
        '--patch-overlap-ratio',
        type=float,
        default=0.25,
        help='Patch overlap ratio used only in slice mode.')
    parser.add_argument(
        '--merge-iou-thr',
        type=float,
        default=0.5,
        help='IoU threshold for merging sliced predictions.')
    parser.add_argument(
        '--slice-batch-size',
        type=int,
        default=4,
        help='Batch size for patch inference in slice mode.')
    parser.add_argument(
        '--max-images',
        type=int,
        default=0,
        help='Only evaluate the first N images. 0 means all images.')
    parser.add_argument(
        '--out-dir',
        default='work_dirs_gravel_big/high_recall_ablation',
        help='Directory to write summary JSON files.')
    parser.add_argument(
        '--tag',
        default='',
        help='Optional tag appended to the summary file name.')
    return parser.parse_args()


def unwrap_dataset_cfg(dataset_cfg):
    while 'dataset' in dataset_cfg:
        dataset_cfg = dataset_cfg['dataset']
    return dataset_cfg


def update_resize_scale(pipeline, scale):
    for transform in pipeline:
        if transform.get('type') == 'Resize':
            transform['scale'] = (scale, scale)
            return
    raise ValueError('Resize transform not found in test pipeline.')


def build_model(cfg_path, checkpoint, args):
    cfg = Config.fromfile(cfg_path)

    if 'init_cfg' in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None

    dataset_cfg = unwrap_dataset_cfg(cfg.test_dataloader.dataset)
    dataset_cfg.ann_file = args.ann_file
    dataset_cfg.data_root = ''
    dataset_cfg.data_prefix = dict(img=args.img_root + '/')
    dataset_cfg.test_mode = True
    update_resize_scale(dataset_cfg.pipeline, args.scale)

    cfg.test_evaluator.ann_file = args.ann_file
    cfg.model.test_cfg.rcnn.max_per_img = args.max_per_img
    cfg.model.test_cfg.rcnn.score_thr = args.score_thr
    cfg.model.test_cfg.rcnn.nms = dict(type='nms', iou_threshold=args.rcnn_nms_iou)
    cfg.model.test_cfg.rpn.nms_pre = args.rpn_nms_pre
    cfg.model.test_cfg.rpn.max_per_img = args.rpn_max_per_img
    cfg.model.test_cfg.rpn.nms = dict(type='nms', iou_threshold=args.rpn_nms_iou)

    return init_detector(cfg, checkpoint, device=args.device, cfg_options={})


def build_metric(ann_file, classes):
    metric = CocoMetric(ann_file=ann_file, metric='segm', collect_device='cpu')
    metric.dataset_meta = {'classes': tuple(classes)}
    return metric


def run_slice_inference(model, img_path, args):
    image = mmcv.imread(img_path)
    height, width = image.shape[:2]
    sliced = slice_image(
        image,
        slice_height=args.patch_size,
        slice_width=args.patch_size,
        auto_slice_resolution=False,
        overlap_height_ratio=args.patch_overlap_ratio,
        overlap_width_ratio=args.patch_overlap_ratio)

    slice_results = []
    start = 0
    while start < len(sliced):
        end = min(start + args.slice_batch_size, len(sliced))
        batch_images = sliced.images[start:end]
        batch_results = inference_detector(model, batch_images)
        if not isinstance(batch_results, list):
            batch_results = [batch_results]
        slice_results.extend(batch_results)
        start = end

    merged = merge_results_by_nms(
        slice_results,
        sliced.starting_pixels,
        src_image_shape=(height, width),
        nms_cfg=dict(type='nms', iou_threshold=args.merge_iou_thr))
    return merged, (height, width)


def run_full_inference(model, img_path):
    result = inference_detector(model, img_path)
    if isinstance(result, list):
        result = result[0]
    image = mmcv.imread(img_path)
    return result, image.shape[:2]


def summarize_metrics(metrics):
    keys = [
        'coco/segm_mAP', 'coco/segm_mAP_50', 'coco/segm_mAP_75',
        'coco/segm_mAP_s', 'coco/segm_mAP_m', 'coco/segm_mAP_l'
    ]
    return {key: float(metrics[key]) for key in keys if key in metrics}


def to_metric_sample(result, img_id, ori_shape):
    return dict(
        img_id=img_id,
        ori_shape=ori_shape,
        pred_instances=result.pred_instances)


def main():
    args = parse_args()

    model = build_model(args.config, args.checkpoint, args)
    classes = model.dataset_meta['classes']
    metric = build_metric(args.ann_file, classes)
    coco_api = metric._coco_api
    img_ids = coco_api.get_img_ids()
    if args.max_images > 0:
        img_ids = img_ids[:args.max_images]
    img_infos = coco_api.load_imgs(img_ids)

    logger = MMLogger.get_current_instance()
    logger.info(
        'Running %s inference on %d images with scale=%d, max_per_img=%d, '
        'rcnn_nms_iou=%.2f, rpn_nms_pre=%d, rpn_max_per_img=%d, '
        'rpn_nms_iou=%.2f', args.mode, len(img_infos), args.scale,
        args.max_per_img, args.rcnn_nms_iou, args.rpn_nms_pre,
        args.rpn_max_per_img, args.rpn_nms_iou)

    progress_bar = ProgressBar(len(img_infos))
    for img_info in img_infos:
        img_path = os.path.join(args.img_root, img_info['file_name'])
        if args.mode == 'slice':
            result, ori_shape = run_slice_inference(model, img_path, args)
        else:
            result, ori_shape = run_full_inference(model, img_path)

        metric.process({}, [to_metric_sample(result, img_info['id'], ori_shape)])
        progress_bar.update()

    metrics = metric.evaluate(size=len(img_infos))
    summary = summarize_metrics(metrics)

    config_name = Path(args.config).stem
    tag = args.tag or f'{args.mode}_scale{args.scale}_k{args.max_per_img}'
    mkdir_or_exist(args.out_dir)
    out_path = Path(args.out_dir) / f'{config_name}_{tag}.json'
    out_payload = dict(
        config=args.config,
        checkpoint=args.checkpoint,
        ann_file=args.ann_file,
        img_root=args.img_root,
        mode=args.mode,
        scale=args.scale,
        score_thr=args.score_thr,
        max_per_img=args.max_per_img,
        rcnn_nms_iou=args.rcnn_nms_iou,
        rpn_nms_pre=args.rpn_nms_pre,
        rpn_max_per_img=args.rpn_max_per_img,
        rpn_nms_iou=args.rpn_nms_iou,
        patch_size=args.patch_size,
        patch_overlap_ratio=args.patch_overlap_ratio,
        merge_iou_thr=args.merge_iou_thr,
        slice_batch_size=args.slice_batch_size,
        max_images=args.max_images,
        metrics=summary)
    out_path.write_text(json.dumps(out_payload, indent=2), encoding='utf-8')

    print(json.dumps(out_payload, indent=2))


if __name__ == '__main__':
    main()
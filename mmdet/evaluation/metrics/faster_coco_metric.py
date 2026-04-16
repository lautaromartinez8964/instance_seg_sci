import itertools
import os.path as osp
import tempfile
from collections import OrderedDict
from typing import Dict, Optional

import numpy as np
from mmengine.fileio import get_local_path, load
from mmengine.logging import MMLogger
from terminaltables import AsciiTable

from mmdet.registry import METRICS

from .coco_metric import CocoMetric

try:
    from faster_coco_eval import COCO as FasterCOCO
    from faster_coco_eval import COCOeval_faster
except ImportError:  # pragma: no cover
    FasterCOCO = None
    COCOeval_faster = None


@METRICS.register_module()
class FasterCocoMetric(CocoMetric):
    """COCO metric backed by faster-coco-eval.

    This avoids the pycocotools evaluation path that can segfault in some
    environments when evaluating instance masks.
    """

    def __init__(self, *args, **kwargs) -> None:
        if FasterCOCO is None or COCOeval_faster is None:
            raise ImportError(
                'faster-coco-eval is required for FasterCocoMetric. '
                'Install it with "pip install faster-coco-eval".')
        ann_file = kwargs.get('ann_file', None)
        sort_categories = kwargs.get('sort_categories', False)
        super().__init__(*args, **kwargs)

        if ann_file is not None:
            with get_local_path(
                    ann_file, backend_args=self.backend_args) as local_path:
                self._coco_api = FasterCOCO(local_path)
                if sort_categories:
                    cats = self._coco_api.cats
                    sorted_cats = {i: cats[i] for i in sorted(cats)}
                    self._coco_api.cats = sorted_cats
                    categories = self._coco_api.dataset['categories']
                    sorted_categories = sorted(
                        categories, key=lambda item: item['id'])
                    self._coco_api.dataset['categories'] = sorted_categories

    def compute_metrics(self, results: list) -> Dict[str, float]:
        logger: MMLogger = MMLogger.get_current_instance()

        gts, preds = zip(*results)

        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix

        if self._coco_api is None:
            logger.info('Converting ground truth to coco format...')
            coco_json_path = self.gt_to_coco_json(
                gt_dicts=gts, outfile_prefix=outfile_prefix)
            self._coco_api = FasterCOCO(coco_json_path)

        if self.cat_ids is None:
            self.cat_ids = self._coco_api.get_cat_ids(
                cat_names=self.dataset_meta['classes'])
        if self.img_ids is None:
            self.img_ids = self._coco_api.get_img_ids()

        result_files = self.results2json(preds, outfile_prefix)

        eval_results = OrderedDict()
        if self.format_only:
            logger.info(f'results are saved in {osp.dirname(outfile_prefix)}')
            return eval_results

        coco_metric_names = {
            'mAP': 0,
            'mAP_50': 1,
            'mAP_75': 2,
            'mAP_s': 3,
            'mAP_m': 4,
            'mAP_l': 5,
            'AR@100': 6,
            'AR@300': 7,
            'AR@1000': 8,
            'AR_s@1000': 9,
            'AR_m@1000': 10,
            'AR_l@1000': 11,
        }

        for metric in self.metrics:
            logger.info(f'Evaluating {metric}...')

            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    preds, self.proposal_nums, self.iou_thrs, logger=logger)
                log_msg = []
                for i, num in enumerate(self.proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                logger.info(''.join(log_msg))
                continue

            iou_type = 'bbox' if metric == 'proposal' else metric
            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')

            try:
                predictions = load(result_files[metric])
                if iou_type == 'segm':
                    for item in predictions:
                        item.pop('bbox', None)
                coco_dt = self._coco_api.loadRes(predictions)
            except IndexError:
                logger.error('The testing results of the whole dataset is empty.')
                break

            coco_eval = COCOeval_faster(
                self._coco_api,
                coco_dt,
                iouType=iou_type,
                print_function=logger.info)
            coco_eval.params.catIds = self.cat_ids
            coco_eval.params.imgIds = self.img_ids
            coco_eval.params.maxDets = list(self.proposal_nums)
            coco_eval.params.iouThrs = self.iou_thrs

            metric_items = self.metric_items
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(
                            f'metric item "{metric_item}" is not supported')

            if metric == 'proposal':
                coco_eval.params.useCats = 0
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                if metric_items is None:
                    metric_items = [
                        'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                        'AR_m@1000', 'AR_l@1000'
                    ]
                for item in metric_items:
                    eval_results[item] = float(
                        f'{coco_eval.stats[coco_metric_names[item]]:.3f}')
                continue

            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            if self.classwise:
                precisions = coco_eval.eval['precision']
                assert len(self.cat_ids) == precisions.shape[2]

                results_per_category = []
                for idx, cat_id in enumerate(self.cat_ids):
                    row = []
                    nm = self._coco_api.loadCats(cat_id)[0]
                    precision = precisions[:, :, idx, 0, -1]
                    precision = precision[precision > -1]
                    ap = np.mean(precision) if precision.size else float('nan')
                    row.append(f'{nm["name"]}')
                    row.append(f'{round(ap, 3)}')
                    eval_results[f'{nm["name"]}_precision'] = round(ap, 3)

                    for iou in [0, 5]:
                        precision = precisions[iou, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        ap = np.mean(precision) if precision.size else float('nan')
                        row.append(f'{round(ap, 3)}')

                    for area in [1, 2, 3]:
                        precision = precisions[:, :, idx, area, -1]
                        precision = precision[precision > -1]
                        ap = np.mean(precision) if precision.size else float('nan')
                        row.append(f'{round(ap, 3)}')
                    results_per_category.append(tuple(row))

                num_columns = len(results_per_category[0])
                results_flatten = list(itertools.chain(*results_per_category))
                headers = [
                    'category', 'mAP', 'mAP_50', 'mAP_75', 'mAP_s',
                    'mAP_m', 'mAP_l'
                ]
                results_2d = itertools.zip_longest(*[
                    results_flatten[i::num_columns]
                    for i in range(num_columns)
                ])
                table_data = [headers]
                table_data += [result for result in results_2d]
                table = AsciiTable(table_data)
                logger.info('\n' + table.table)

            if metric_items is None:
                metric_items = [
                    'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                ]

            for metric_item in metric_items:
                key = f'{metric}_{metric_item}'
                val = coco_eval.stats[coco_metric_names[metric_item]]
                eval_results[key] = float(f'{round(val, 3)}')

            ap = coco_eval.stats[:6]
            logger.info(f'{metric}_mAP_copypaste: {ap[0]:.3f} '
                        f'{ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                        f'{ap[4]:.3f} {ap[5]:.3f}')

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results
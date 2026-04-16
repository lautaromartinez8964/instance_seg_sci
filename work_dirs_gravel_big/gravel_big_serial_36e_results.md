# Gravel Big Serial 36e Results

该文件由串行训练脚本自动维护。

## Mask R-CNN R50

- config: configs/gravel_big/mask_rcnn_r50_fpn_36e_gravel_big.py
- work_dir: work_dirs_gravel_big/mask_rcnn_r50_fpn_36e_gravel_big
- best_ckpt: /home/yxy18034962/projects/mmdetection/work_dirs_gravel_big/mask_rcnn_r50_fpn_36e_gravel_big/best_coco_segm_mAP_epoch_31.pth
- eval_log: /home/yxy18034962/projects/mmdetection/work_dirs_gravel_big/mask_rcnn_r50_fpn_36e_gravel_big/test_eval.log

```text
04/15 21:30:11 - mmengine - INFO -  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.264
04/15 21:30:11 - mmengine - INFO -  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.418
04/15 21:30:11 - mmengine - INFO -  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.300
04/15 21:30:11 - mmengine - INFO -  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.209
04/15 21:30:11 - mmengine - INFO -  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.535
04/15 21:30:11 - mmengine - INFO -  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.699
04/15 21:30:11 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.296
04/15 21:30:11 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.296
04/15 21:30:11 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.296
04/15 21:30:11 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.236
04/15 21:30:11 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.592
04/15 21:30:11 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.735
04/15 21:30:11 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.440
04/15 21:30:11 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.339
```

## Official VMamba 2292

- config: configs/gravel_big/mask_rcnn_vmamba_official_2292_fpn_36e_gravel_big.py
- work_dir: work_dirs_gravel_big/mask_rcnn_vmamba_official_2292_fpn_36e_gravel_big
- best_ckpt: /home/yxy18034962/projects/mmdetection/work_dirs_gravel_big/mask_rcnn_vmamba_official_2292_fpn_36e_gravel_big/best_coco_segm_mAP_epoch_27.pth
- eval_log: /home/yxy18034962/projects/mmdetection/work_dirs_gravel_big/mask_rcnn_vmamba_official_2292_fpn_36e_gravel_big/test_eval.log

```text
04/16 01:01:54 - mmengine - INFO -  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.274
04/16 01:01:54 - mmengine - INFO -  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.428
04/16 01:01:54 - mmengine - INFO -  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.309
04/16 01:01:54 - mmengine - INFO -  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.216
04/16 01:01:54 - mmengine - INFO -  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.547
04/16 01:01:54 - mmengine - INFO -  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.696
04/16 01:01:54 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.303
04/16 01:01:54 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.303
04/16 01:01:54 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.303
04/16 01:01:54 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.244
04/16 01:01:54 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.596
04/16 01:01:54 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.729
04/16 01:01:54 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.440
04/16 01:01:54 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.348
```

## RS-LightMamba S4 GlobalAttn

- config: configs/gravel_big/mask_rcnn_rs_lightmamba_s4_global_attn_fpn_36e_gravel_big.py
- work_dir: work_dirs_gravel_big/mask_rcnn_rs_lightmamba_s4_global_attn_fpn_36e_gravel_big
- best_ckpt: /home/yxy18034962/projects/mmdetection/work_dirs_gravel_big/mask_rcnn_rs_lightmamba_s4_global_attn_fpn_36e_gravel_big/best_coco_segm_mAP_epoch_25.pth
- eval_log: /home/yxy18034962/projects/mmdetection/work_dirs_gravel_big/mask_rcnn_rs_lightmamba_s4_global_attn_fpn_36e_gravel_big/test_eval.log

```text
04/16 05:44:15 - mmengine - INFO -  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.271
04/16 05:44:15 - mmengine - INFO -  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.419
04/16 05:44:15 - mmengine - INFO -  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.310
04/16 05:44:15 - mmengine - INFO -  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.215
04/16 05:44:15 - mmengine - INFO -  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.548
04/16 05:44:15 - mmengine - INFO -  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.684
04/16 05:44:15 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.300
04/16 05:44:15 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.300
04/16 05:44:15 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.300
04/16 05:44:15 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.241
04/16 05:44:15 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.596
04/16 05:44:15 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.717
04/16 05:44:15 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.438
04/16 05:44:15 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.342
```

## RS-LightMamba S4 GlobalAttn HF-FPN

- config: configs/gravel_big/mask_rcnn_rs_lightmamba_s4_global_attn_hf_fpn_36e_gravel_big.py
- work_dir: work_dirs_gravel_big/mask_rcnn_rs_lightmamba_s4_global_attn_hf_fpn_36e_gravel_big
- best_ckpt: /home/yxy18034962/projects/mmdetection/work_dirs_gravel_big/mask_rcnn_rs_lightmamba_s4_global_attn_hf_fpn_36e_gravel_big/best_coco_segm_mAP_epoch_28.pth
- eval_log: /home/yxy18034962/projects/mmdetection/work_dirs_gravel_big/mask_rcnn_rs_lightmamba_s4_global_attn_hf_fpn_36e_gravel_big/test_eval.log

```text
04/16 10:34:30 - mmengine - INFO -  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.274
04/16 10:34:30 - mmengine - INFO -  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.427
04/16 10:34:30 - mmengine - INFO -  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.312
04/16 10:34:30 - mmengine - INFO -  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.219
04/16 10:34:30 - mmengine - INFO -  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.544
04/16 10:34:30 - mmengine - INFO -  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.692
04/16 10:34:30 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.303
04/16 10:34:30 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.303
04/16 10:34:30 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.303
04/16 10:34:30 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.245
04/16 10:34:30 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.591
04/16 10:34:30 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.722
04/16 10:34:30 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.441
04/16 10:34:30 - mmengine - INFO -  Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.348
```


"""Normalized Gaussian Wasserstein Distance Loss for bbox regression.

IMPORTANT: This loss operates on DECODED (absolute xyxy) bounding boxes.
It must NOT be used with the default encoded delta targets.

Two usage modes:
  Mode A (recommended): Set reg_decoded_bbox=True in the head config.
    This tells the head to decode predictions & targets before passing
    them to the loss. Works for both RPN and RoI heads.

  Mode B: This loss auto-decodes internally using provided priors.
    Set reg_decoded_bbox=True and the head will handle decoding.
    Alternatively, if reg_decoded_bbox=False, the loss will receive
    deltas — which is mathematically wrong for NWD.

Why decoded boxes?
  NWD measures the Gaussian Wasserstein Distance between two spatial
  distributions. Encoded deltas (dx, dy, dw, dh) are NOT spatial
  coordinates — computing NWD on them has no geometric meaning.
  Only absolute xyxy boxes define meaningful 2D Gaussians.

Weight handling:
  MMDet passes bbox_weights as (N, 4) tensors. Since NWD produces
  a per-box scalar (not per-coordinate), we reduce weight to (N,)
  before applying it.

Usage in config (Mode A, for RoI head):
    roi_head=dict(
        bbox_head=dict(
            reg_decoded_bbox=True,
            loss_bbox=dict(type='NWDLoss', loss_weight=5.0, constant=8.0)))

Usage in config (Mode A, for RPN head):
    rpn_head=dict(
        reg_decoded_bbox=True,
        loss_bbox=dict(type='NWDLoss', loss_weight=5.0, constant=8.0))
"""
import torch
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.models.task_modules.assigners.nwd_calculator import bbox_to_gaussian


@MODELS.register_module()
class NWDLoss:
    """NWD-based bounding box regression loss.

    Instead of computing L1/SmoothL1/GIoU on box coordinates or deltas,
    we compute the Gaussian Wasserstein Distance between predicted and
    target boxes (both in absolute xyxy format), then convert to a loss:
        L = 1 - NWD

    This provides:
    1. Smooth gradients even for non-overlapping boxes
    2. Scale-balanced gradients (tiny and large objects contribute
       comparable gradient magnitudes)
    3. Consistent optimization objective with NWD-based assigner

    CRITICAL: Both pred_bboxes and gt_bboxes must be in absolute xyxy
    format (decoded), NOT encoded deltas. Use reg_decoded_bbox=True
    in the corresponding head.

    Args:
        constant: Normalization constant C for NWD. Default: 8.0.
        loss_weight: Loss weight. Default: 5.0 (NWD values are typically
            smaller than L1 loss, so we upweight).
        reduction: Reduction method. Default: 'mean'.
    """

    def __init__(self,
                 constant: float = 8.0,
                 loss_weight: float = 5.0,
                 reduction: str = 'mean'):
        self.constant = constant
        self.loss_weight = loss_weight
        self.reduction = reduction

    def __call__(self,
                 pred_bboxes: Tensor,
                 gt_bboxes: Tensor,
                 weight=None,
                 avg_factor=None,
                 **kwargs) -> Tensor:
        """Compute NWD loss.

        Args:
            pred_bboxes: (N, 4) predicted boxes in absolute xyxy format.
                MUST be decoded (not deltas). Use reg_decoded_bbox=True.
            gt_bboxes: (N, 4) ground truth boxes in absolute xyxy format.
            weight: Per-sample weight. Can be (N,) or (N, 4).
                If (N, 4), it will be reduced to (N,) by taking the mean
                across the coordinate dimension, since NWD is a box-level
                (not coordinate-level) metric.
            avg_factor: Average factor for reduction.

        Returns:
            Loss value (scalar).
        """
        # Safety check: if inputs look like deltas (small values near 0),
        # warn the user. This is a heuristic — deltas typically have
        # values in [-2, 2] while absolute coords are >> 1.
        if pred_bboxes.abs().max() < 5.0 and pred_bboxes.numel() > 0:
            import warnings
            warnings.warn(
                'NWDLoss received very small bbox values (max abs < 5). '
                'This may indicate encoded deltas instead of decoded xyxy '
                'boxes. Please set reg_decoded_bbox=True in the head config.')

        # Clamp to avoid degenerate boxes
        pred_bboxes = pred_bboxes.clamp(min=0)
        gt_bboxes = gt_bboxes.clamp(min=0)

        mu_pred, sigma_pred = bbox_to_gaussian(pred_bboxes)
        mu_gt, sigma_gt = bbox_to_gaussian(gt_bboxes)

        # Wasserstein distance squared (aligned/pairwise)
        center_dist = (mu_pred - mu_gt).pow(2).sum(dim=-1)  # (N,)
        sigma_dist = (sigma_pred - sigma_gt).pow(2).sum(dim=-1)  # (N,)
        w2 = center_dist + sigma_dist  # (N,)

        # NWD similarity
        nwd = torch.exp(-torch.sqrt(w2 + 1e-7) / self.constant)  # (N,)

        # Loss = 1 - NWD (maximize similarity = minimize loss)
        loss = 1.0 - nwd  # (N,)

        # Handle weight: MMDet passes (N, 4) weights, but NWD is (N,)
        if weight is not None:
            if weight.dim() > 1 and weight.size(-1) > 1:
                # Reduce (N, 4) → (N,) by taking mean across coordinates
                # This is correct because all 4 coordinate weights are 1.0
                # for positive samples (see anchor_head.py:284, bbox_head.py:219)
                weight = weight.mean(dim=-1)
            loss = loss * weight

        # Reduction
        if self.reduction == 'mean':
            if avg_factor is not None:
                loss = loss.sum() / avg_factor
            else:
                loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss * self.loss_weight

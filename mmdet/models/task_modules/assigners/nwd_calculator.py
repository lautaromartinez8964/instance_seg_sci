"""Normalized Gaussian Wasserstein Distance (NWD) Calculator.

Reference:
    A Simple Baseline for Tiny Object Detection with Normalized
    Gaussian Wasserstein Distance (AAAI 2021)

The key insight: modeling bounding boxes as 2D Gaussian distributions
and using Wasserstein distance instead of IoU provides smooth gradients
even for non-overlapping tiny objects.

Physical intuition:
    - IoU is a "binary" metric: overlap or no overlap
    - NWD is a "field" metric: even non-overlapping boxes have
      non-zero similarity via their Gaussian tails
    - This is critical for tiny objects where even 1-pixel offset
      can cause IoU to drop to 0
"""
import torch
import torch.nn.functional as F
from torch import Tensor

from mmdet.registry import TASK_UTILS


def bbox_to_gaussian(bboxes: Tensor) -> tuple:
    """Convert xyxy bounding boxes to 2D Gaussian parameters.

    Args:
        bboxes: Shape (N, 4) in xyxy format.

    Returns:
        mu: Shape (N, 2) - Gaussian centers.
        sigma: Shape (N, 2) - Gaussian standard deviations (w/4, h/4).
    """
    cx = (bboxes[:, 0] + bboxes[:, 2]) / 2
    cy = (bboxes[:, 1] + bboxes[:, 3]) / 2
    w = bboxes[:, 2] - bboxes[:, 0]
    h = bboxes[:, 3] - bboxes[:, 1]
    # Clamp to avoid zero-size boxes
    w = w.clamp(min=1e-7)
    h = h.clamp(min=1e-7)
    mu = torch.stack([cx, cy], dim=-1)  # (N, 2)
    sigma = torch.stack([w / 4, h / 4], dim=-1)  # (N, 2)
    return mu, sigma


def _compute_nwd_matrix(bboxes1: Tensor,
                        bboxes2: Tensor,
                        constant: float) -> Tensor:
    """Compute NWD similarity matrix between two bbox sets.

    Args:
        bboxes1: (M, 4) in xyxy format.
        bboxes2: (N, 4) in xyxy format.
        constant: Normalization constant C.

    Returns:
        nwd: (M, N) similarity matrix, values in (0, 1].
    """
    mu1, sigma1 = bbox_to_gaussian(bboxes1)  # (M, 2)
    mu2, sigma2 = bbox_to_gaussian(bboxes2)  # (N, 2)

    # Center distance: ||mu1 - mu2||^2
    center_dist = (mu1.unsqueeze(1) - mu2.unsqueeze(0)).pow(2).sum(dim=-1)
    # Sigma distance: ||sigma1 - sigma2||_F^2
    sigma_dist = (sigma1.unsqueeze(1) - sigma2.unsqueeze(0)).pow(2).sum(dim=-1)
    # Wasserstein distance squared
    w2 = center_dist + sigma_dist  # (M, N)
    return torch.exp(-torch.sqrt(w2 + 1e-7) / constant)


def _compute_nwd_aligned(bboxes1: Tensor,
                         bboxes2: Tensor,
                         constant: float) -> Tensor:
    """Compute pairwise NWD similarity.

    Args:
        bboxes1: (N, 4) in xyxy format.
        bboxes2: (N, 4) in xyxy format.
        constant: Normalization constant C.

    Returns:
        nwd: (N,) similarity vector, values in (0, 1].
    """
    mu1, sigma1 = bbox_to_gaussian(bboxes1)
    mu2, sigma2 = bbox_to_gaussian(bboxes2)
    center_dist = (mu1 - mu2).pow(2).sum(dim=-1)
    sigma_dist = (sigma1 - sigma2).pow(2).sum(dim=-1)
    w2 = center_dist + sigma_dist
    return torch.exp(-torch.sqrt(w2 + 1e-7) / constant)


@TASK_UTILS.register_module()
class BboxOverlaps2D_NWD:
    """NWD-based overlap calculator, drop-in replacement for BboxOverlaps2D.

    This calculator can be used in MaxIoUAssigner by simply replacing:
        iou_calculator=dict(type='BboxOverlaps2D')
    with:
        iou_calculator=dict(type='BboxOverlaps2D_NWD', constant=8.0)

    The output is a similarity matrix in (0, 1] — same semantics as IoU,
    so MaxIoUAssigner's threshold logic works directly.

    Supports both 'iou' and 'iof' modes for full API compatibility.
    For NWD, both modes produce the same result because NWD is a
    symmetric distance metric (unlike IoU vs IoF which differ in
    their denominator).

    Args:
        constant: Normalization constant C for NWD computation.
            Recommended values:
            - Tiny objects (<16px): C = 2.0 ~ 4.0
            - Small objects (<32px): C = 4.0 ~ 8.0
            - Mixed scales: C = 8.0 ~ 12.0
            Default: 8.0 (works well for most remote sensing scenarios).
    """

    def __init__(self, constant: float = 8.0):
        self.constant = constant

    def __call__(self, bboxes1, bboxes2, mode='iou', is_aligned=False):
        """Compute NWD similarity matrix.

        Args:
            bboxes1: (M, 4) or (M, 5) in xyxy format.
            bboxes2: (N, 4) or (N, 5) in xyxy format.
            mode: 'iou' or 'iof'. For NWD, both produce the same result
                because Wasserstein distance is symmetric. Kept for API
                compatibility with BboxOverlaps2D and MaxIoUAssigner's
                ignore branch which calls mode='iof'.
            is_aligned: If True, compute pairwise (M,) instead of (M, N).

        Returns:
            Similarity matrix (M, N) or (M,), values in (0, 1].
        """
        # Handle 5-dim bboxes (with scores)
        if bboxes1.size(-1) == 5:
            bboxes1 = bboxes1[..., :4]
        if bboxes2.size(-1) == 5:
            bboxes2 = bboxes2[..., :4]

        # Note: mode is intentionally ignored. NWD is a symmetric
        # distance metric — unlike IoU/IoF which differ by denominator
        # (intersection over union vs intersection over foreground area),
        # NWD computes the same Gaussian Wasserstein Distance regardless
        # of which box set is considered "foreground". This is correct
        # behavior for the MaxIoUAssigner ignore branch (mode='iof'),
        # where the goal is to measure similarity between anchors and
        # ignored regions — NWD does this symmetrically.
        if is_aligned:
            return _compute_nwd_aligned(bboxes1, bboxes2, self.constant)
        else:
            return _compute_nwd_matrix(bboxes1, bboxes2, self.constant)

    def __repr__(self):
        return f'{self.__class__.__name__}(constant={self.constant})'

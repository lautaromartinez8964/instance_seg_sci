import sys
import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '.')

MODULE_PATH = Path(
    'mmdet/models/backbones/rs_lightmamba/fg_ig_scan.py').resolve()
SPEC = importlib.util.spec_from_file_location('fg_ig_scan_module', MODULE_PATH)
FG_IG_SCAN_MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(FG_IG_SCAN_MODULE)

SCAN_MODULE_PATH = Path(
    'mmdet/models/backbones/rs_lightmamba/ig_cross_scan.py').resolve()
SCAN_SPEC = importlib.util.spec_from_file_location(
    'ig_cross_scan_module', SCAN_MODULE_PATH)
IG_CROSS_SCAN_MODULE = importlib.util.module_from_spec(SCAN_SPEC)
assert SCAN_SPEC.loader is not None
SCAN_SPEC.loader.exec_module(IG_CROSS_SCAN_MODULE)

FGIGScan = FG_IG_SCAN_MODULE.FGIGScan
ig_cross_merge_fn = IG_CROSS_SCAN_MODULE.ig_cross_merge_fn
ig_cross_scan_fn = IG_CROSS_SCAN_MODULE.ig_cross_scan_fn
restore_ig_path = IG_CROSS_SCAN_MODULE.restore_ig_path


def _device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def test_shape_preservation():
    device = _device()
    batch_size, channels, height, width = 2, 32, 16, 16
    x = torch.randn(batch_size, channels, height, width, device=device)
    scan = FGIGScan(d_inner=channels, region_size=4, guidance_scale=0.1).to(device)

    x_mod, order, gh, gw, _ = scan.compute_region_order(x, descending=True)
    xs, scan_info = ig_cross_scan_fn(x_mod, order, gh, gw)
    y = ig_cross_merge_fn(xs, scan_info)

    assert xs.shape == (batch_size, 4, channels, height * width)
    assert y.shape == x.shape


def test_identity_order_matches_official_scan0():
    device = _device()
    batch_size, channels, height, width = 1, 8, 16, 16
    x = torch.randn(batch_size, channels, height, width, device=device)
    gh = gw = 4
    order = torch.arange(gh * gw, device=device).view(1, -1)

    xs, scan_info = ig_cross_scan_fn(x, order, gh, gw)
    restored = restore_ig_path(xs[:, 0], scan_info, path_type='row', reverse=False)
    assert torch.allclose(restored, x, atol=1e-6)

    restored_rev = restore_ig_path(xs[:, 2], scan_info, path_type='row', reverse=True)
    assert torch.allclose(restored_rev, x, atol=1e-6)


def test_gradient_flow_to_fg_head():
    device = _device()
    scan = FGIGScan(d_inner=32, region_size=4, guidance_scale=0.1).to(device)
    x = torch.randn(2, 32, 16, 16, device=device, requires_grad=True)
    x_mod, order, gh, gw, _ = scan.compute_region_order(x, descending=True)
    xs, scan_info = ig_cross_scan_fn(x_mod, order, gh, gw)
    y = ig_cross_merge_fn(xs, scan_info)
    y.sum().backward()

    assert x.grad is not None
    assert scan._guidance_scale_raw.grad is not None
    assert scan.fg_head.pw_out.weight.grad is not None


def test_foreground_supervision_loss_is_generated():
    device = _device()
    scan = FGIGScan(
        d_inner=16,
        region_size=4,
        guidance_scale=0.1,
        fg_loss_weight=0.2).to(device)
    x = torch.randn(2, 16, 16, 16, device=device)
    fg_target = torch.zeros(2, 1, 32, 32, device=device)
    fg_target[:, :, 8:24, 8:24] = 1.0
    scan.set_fg_target(fg_target)

    scan.compute_region_order(x, descending=True)

    assert scan.last_fg_loss is not None
    assert scan.last_fg_loss.item() > 0


def test_area_downsample_preserves_tiny_foreground():
    device = _device()
    scan = FGIGScan(
        d_inner=16,
        region_size=4,
        guidance_scale=0.1,
        fg_loss_weight=1.0).to(device)
    importance = torch.full((1, 1, 8, 8), 0.1, device=device)
    fg_target = torch.zeros(1, 1, 32, 32, device=device)
    fg_target[:, :, 3, 3] = 1.0
    scan.set_fg_target(fg_target)

    loss = scan._compute_fg_loss(importance)
    expected_target = torch.zeros_like(importance)
    expected_target[:, :, 0, 0] = 1.0
    expected = F.binary_cross_entropy(importance.float(), expected_target.float())

    assert loss is not None
    assert torch.allclose(loss, expected, atol=1e-6)


def test_predict_importance_updates_runtime_state():
    device = _device()
    scan = FGIGScan(d_inner=16, region_size=4, guidance_scale=0.1).to(device)
    x = torch.randn(1, 16, 8, 8, device=device)

    importance = scan.predict_importance(x)

    assert importance.shape == (1, 1, 8, 8)
    assert scan.last_importance is not None
    assert scan.last_order is None


def test_foreground_head_supports_group_norm():
    device = _device()
    scan = FGIGScan(
        d_inner=32,
        region_size=4,
        guidance_scale=0.1,
        fg_norm_type='gn',
        fg_gn_groups=8).to(device)

    assert isinstance(scan.fg_head.bn1, nn.GroupNorm)
    assert isinstance(scan.fg_head.bn2, nn.GroupNorm)
    assert isinstance(scan.fg_head.bn3, nn.GroupNorm)
    assert isinstance(scan.fg_head.bn4, nn.GroupNorm)


if __name__ == '__main__':
    test_shape_preservation()
    test_identity_order_matches_official_scan0()
    test_gradient_flow_to_fg_head()
    test_foreground_supervision_loss_is_generated()
    test_area_downsample_preserves_tiny_foreground()
    test_predict_importance_updates_runtime_state()
    test_foreground_head_supports_group_norm()
    print('ALL TESTS PASSED')
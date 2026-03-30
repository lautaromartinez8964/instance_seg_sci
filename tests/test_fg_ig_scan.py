import sys
import importlib.util
from pathlib import Path

import torch

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


if __name__ == '__main__':
    test_shape_preservation()
    test_identity_order_matches_official_scan0()
    test_gradient_flow_to_fg_head()
    print('ALL TESTS PASSED')
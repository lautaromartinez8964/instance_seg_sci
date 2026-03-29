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

FGIGScan = FG_IG_SCAN_MODULE.FGIGScan
get_region_order = FG_IG_SCAN_MODULE.get_region_order
region_inv_permute_2d = FG_IG_SCAN_MODULE.region_inv_permute_2d
region_permute_2d = FG_IG_SCAN_MODULE.region_permute_2d


def _device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def test_roundtrip():
    device = _device()
    x = torch.randn(2, 16, 16, 16, device=device)
    order = torch.stack([torch.randperm(16) for _ in range(2)]).to(device)
    x_perm = region_permute_2d(x, order, 4, 4)
    x_back = region_inv_permute_2d(x_perm, order, 4, 4)
    assert torch.allclose(x, x_back, atol=1e-6)


def test_constant_score_identity_order():
    scores = torch.ones(2, 8)
    order = get_region_order(scores, descending=True)
    expected = torch.arange(8).view(1, 8).repeat(2, 1)
    assert torch.equal(order.cpu(), expected)


def test_scan_end_to_end_shapes():
    device = _device()
    scan = FGIGScan(d_inner=32, region_size=4).to(device)
    x = torch.randn(2, 32, 16, 16, device=device)
    x_perm = scan.permute_2d(x, descending=True)
    y = scan.inv_permute_2d(x_perm)
    assert x_perm.shape == x.shape
    assert y.shape == x.shape


def test_gradient_flow_to_fg_head():
    device = _device()
    scan = FGIGScan(d_inner=32, region_size=4).to(device)
    x = torch.randn(1, 32, 16, 16, device=device, requires_grad=True)
    out = scan.permute_2d(x, descending=True)
    out.sum().backward()
    assert x.grad is not None
    assert scan.fg_head.pw_conv.weight.grad is not None


if __name__ == '__main__':
    test_roundtrip()
    test_constant_score_identity_order()
    test_scan_end_to_end_shapes()
    test_gradient_flow_to_fg_head()
    print('ALL TESTS PASSED')
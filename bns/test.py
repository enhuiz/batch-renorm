import torch
import traceback
from torch import nn

from .bns import BN, BNRS, BRN


def _assert_close(a, b):
    assert a.isclose(b, 1e-2).all(), (a, b, traceback.print_exc())


def test_bn_1d():
    a = nn.BatchNorm1d(3).train()
    b = BN(3).train()

    assert (a.weight == b.weight).all()
    assert (a.bias == b.bias).all()

    x = torch.randn(3, 3, 3)
    _assert_close(a(x), b(x))

    x = torch.randn(3, 3, 3)
    _assert_close(a(x), b(x))

    a.eval()
    b.eval()

    x = torch.randn(3, 3, 3)
    _assert_close(a(x), b(x))

    x = torch.randn(3, 3, 3)
    _assert_close(a(x), b(x))


def test_bn_2d():
    a = nn.BatchNorm2d(3).train()
    b = BN(3).train()

    assert (a.weight == b.weight).all()
    assert (a.bias == b.bias).all()

    x = torch.randn(3, 3, 3, 3)
    _assert_close(a(x), b(x))

    x = torch.randn(3, 3, 3, 3)
    _assert_close(a(x), b(x))

    a.eval()
    b.eval()

    x = torch.randn(3, 3, 3, 3)
    _assert_close(a(x), b(x))

    x = torch.randn(3, 3, 3, 3)
    _assert_close(a(x), b(x))


def test_brns_brn_1d():
    a = BNRS(3).train()
    b = BRN(3).train()

    assert (a.weight == b.weight).all()
    assert (a.bias == b.bias).all()

    x = torch.randn(3, 3, 3)
    _assert_close(a(x), b(x))

    x = torch.randn(3, 3, 3)
    _assert_close(a(x), b(x))

    a.eval()
    b.eval()

    x = torch.randn(3, 3, 3)
    _assert_close(a(x), b(x))

    x = torch.randn(3, 3, 3)
    _assert_close(a(x), b(x))


if __name__ == "__main__":
    test_bn_1d()
    test_bn_2d()
    test_brns_brn_1d()

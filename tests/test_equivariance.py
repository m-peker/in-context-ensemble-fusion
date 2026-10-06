"""Symmetry properties claimed in the paper (run on CPU, random untrained-but-nonzero weights).

* permuting models  -> fused prediction unchanged, weights permuted accordingly
* permuting classes -> fused prediction permuted accordingly
* permuting context rows -> fused prediction unchanged
* query rows do not influence each other
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from fusionpfn.features import cell_features, class_features  # noqa: E402
from fusionpfn.model import FusionPFN  # noqa: E402

torch.manual_seed(0)
B, n, m, K, C, d = 1, 40, 7, 5, 4, 3


def _model():
    net = FusionPFN(d=32, layers=2, heads=2, residual=True).eval()
    with torch.no_grad():  # zero-initialised heads would make the tests trivial
        for p in net.parameters():
            p.add_(0.1 * torch.randn_like(p))
    return net


def _data():
    P_ctx = torch.softmax(torch.randn(B, n, K, C) * 2, -1)
    P_qry = torch.softmax(torch.randn(B, m, K, C) * 2, -1)
    y = torch.randint(0, C, (B, n))
    X_ctx, X_qry = torch.randn(B, n, d), torch.randn(B, m, d)
    return P_ctx, y, P_qry, X_ctx, X_qry


def _run(net, P_ctx, y, P_qry, X_ctx, X_qry):
    F_ctx, F_qry, _, logP_qry = cell_features(P_ctx, y, P_qry, X_ctx, X_qry)
    G = class_features(P_ctx, y, P_qry)
    with torch.no_grad():
        logp, w = net(F_ctx, F_qry, logP_qry, G_qry=G)
    return logp, w


NET, DATA = _model(), _data()
BASE = _run(NET, *DATA)


def test_model_permutation():
    perm = torch.randperm(K)
    P_ctx, y, P_qry, Xc, Xq = DATA
    logp, w = _run(NET, P_ctx[:, :, perm], y, P_qry[:, :, perm], Xc, Xq)
    assert torch.allclose(logp, BASE[0], atol=1e-4)
    assert torch.allclose(w, BASE[1][:, :, perm], atol=1e-4)


def test_class_permutation():
    perm = torch.randperm(C)
    inv = torch.argsort(perm)
    P_ctx, y, P_qry, Xc, Xq = DATA
    logp, _ = _run(NET, P_ctx[..., perm], inv[y], P_qry[..., perm], Xc, Xq)
    assert torch.allclose(logp, BASE[0][..., perm], atol=1e-4)


def test_context_row_permutation():
    perm = torch.randperm(n)
    P_ctx, y, P_qry, Xc, Xq = DATA
    logp, _ = _run(NET, P_ctx[:, perm], y[:, perm], P_qry, Xc[:, perm], Xq)
    assert torch.allclose(logp, BASE[0], atol=1e-4)


def test_query_rows_independent():
    P_ctx, y, P_qry, Xc, Xq = DATA
    logp, _ = _run(NET, P_ctx, y, P_qry[:, :3], Xc, Xq[:, :3])
    assert torch.allclose(logp, BASE[0][:, :3], atol=1e-4)

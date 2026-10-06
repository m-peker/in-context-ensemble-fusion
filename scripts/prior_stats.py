"""Compare committee statistics of the synthetic prior with REAL committees (dev split only).

Usage: python scripts/prior_stats.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from evaluate import iter_folds  # noqa: E402
from fusionpfn.prior import sample_committee_tasks  # noqa: E402


def stats(P, y):
    """P (n,K,C) numpy, y (n,) -> dict of committee statistics."""
    n, K, C = P.shape
    a = (P.argmax(-1) == y[:, None]).mean(0)
    am = P.argmax(-1)
    agree = (am[:, :, None] == am[:, None, :])[:, ~np.eye(K, dtype=bool)].mean()
    avg_acc = (P.mean(1).argmax(-1) == y).mean()
    oracle = (am == y[:, None]).any(1).mean()
    maxp = P.max(-1).mean()
    # calibration gap of individual models: mean confidence - accuracy
    gap = (P.max(-1).mean(0) - a).mean()
    ll = -np.log(np.clip(P[np.arange(n), :, y], 1e-6, None)).mean(0)  # per model
    avg_ll = -np.log(np.clip(P.mean(1)[np.arange(n), y], 1e-6, None)).mean()
    return {"acc_med": np.median(a), "acc_spread": a.max() - a.min(), "agree": agree,
            "avg-best_acc": avg_acc - a.max(), "oracle-best": oracle - a.max(), "maxp": maxp,
            "overconf": gap, "avg_ll/best_ll": avg_ll / ll.min()}


def summarize(name, rows):
    keys = rows[0].keys()
    print(f"\n{name} (n={len(rows)})  [p10 / p50 / p90]")
    for k in keys:
        v = np.array([r[k] for r in rows])
        print(f"  {k:15s} " + " / ".join(f"{x:7.3f}" for x in np.percentile(v, [10, 50, 90])))


def main():
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    dev = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
    real = []
    for meta, fold, path in iter_folds("dev"):
        if fold != 0:
            continue
        z = np.load(path)
        real.append(stats(z["P_te"], z["y_te"].astype(int)))
    summarize("REAL dev committees (test rows)", real)
    torch.manual_seed(0)
    syn = []
    for C in [2, 2, 2, 3, 4, 6, 10]:
        for K in [4, 8, 8]:
            d = sample_committee_tasks(8, 512, 512, K, C, device=dev)
            for b in range(8):
                syn.append(stats(d["P_qry"][b].cpu().numpy(), d["y_qry"][b].cpu().numpy()))
    summarize("SYNTHETIC prior", syn)


if __name__ == "__main__":
    main()

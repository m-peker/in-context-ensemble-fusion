"""Diagnostic: compare FusionPFN checkpoints with per-task baselines on tasks drawn from the synthetic prior.

Useful to separate model/training issues (FusionPFN loses already on prior tasks) from prior-realism issues
(FusionPFN wins on prior tasks but not on real committees). Also reports the size of the correction term.
Usage: python scripts/diag_synthetic.py <ckpt name> [<ckpt name> ...]   (checkpoints in runs/ckpt/)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from bench.baselines import greedy_ensemble_selection, stacking_lr, arith_mean  # noqa: E402
from fusionpfn.features import cell_features, class_features  # noqa: E402
from fusionpfn.fuse import load_model  # noqa: E402
from fusionpfn.prior import sample_committee_tasks  # noqa: E402


def ll(P, y):
    return -np.log(np.clip(P[np.arange(len(y)), y], 1e-15, None)).mean()


def main():
    torch.manual_seed(123)
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", nargs="+", help="checkpoint names in runs/ckpt/ (without .pt)")
    names = ap.parse_args().ckpt
    dev = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
    models = {n: load_model(os.path.join(ROOT, "runs", "ckpt", f"{n}.pt"), device=dev) for n in names}
    res = {k: [] for k in ["arith", "greedy_ES", "stack_LR", *models]}
    rmag = []
    for n_ctx in [128, 1024]:
        for C in [2, 3, 5]:
            for _ in range(6):
                d = sample_committee_tasks(4, n_ctx, 512, 8, C, device=dev)
                for b in range(4):
                    Pc, yc = d["P_ctx"][b].cpu().numpy(), d["y_ctx"][b].cpu().numpy()
                    Pq, yq = d["P_qry"][b].cpu().numpy(), d["y_qry"][b].cpu().numpy()
                    if len(np.unique(yc)) < C:
                        continue
                    row = {"arith": ll(arith_mean(Pc, yc, Pq), yq),
                           "greedy_ES": ll(greedy_ensemble_selection(Pc, yc, Pq), yq),
                           "stack_LR": ll(stacking_lr(Pc, yc, Pq), yq)}
                    for name, m in models.items():
                        sl = lambda t: t[b:b + 1]
                        F_ctx, F_qry, _, logP = cell_features(sl(d["P_ctx"]), sl(d["y_ctx"]), sl(d["P_qry"]),
                                                              sl(d["X_ctx"]), sl(d["X_qry"]))
                        G = class_features(sl(d["P_ctx"]), sl(d["y_ctx"]), sl(d["P_qry"])) if m.residual else None
                        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                            out = m(F_ctx, F_qry, logP, G_qry=G, return_parts=True)
                        row[name] = ll(out[0].float().exp()[0].cpu().numpy(), yq)
                        if m.residual:
                            pool = torch.einsum("bmk,bmkc->bmc", out[1].float(), logP)
                            pool = pool - pool.mean(-1, keepdim=True)
                            r = out[2].float() - out[2].float().mean(-1, keepdim=True)
                            rmag.append((r.abs().mean() / pool.abs().mean()).item())
                    for k, v in row.items():
                        res[k].append((n_ctx, v / row["arith"]))
    for n_ctx in [128, 1024]:
        print(f"\nsynthetic, n_ctx={n_ctx}: mean log-loss relative to arith_mean")
        for k, v in res.items():
            a = np.array([x for nc, x in v if nc == n_ctx])
            print(f"  {k:12s} {a.mean():.4f}")
    if rmag:
        print(f"\nresidual |r| / |pooled logits| (centered): mean {np.mean(rmag):.4f}")


if __name__ == "__main__":
    main()

"""Merge result CSVs and report ranks, relative log-loss, W/T/L and Holm-corrected Wilcoxon tests.

Usage: python scripts/report.py results/a.csv results/b.csv [--ref FusionPFN:<ckpt name>] [--metric logloss]
Only datasets on which every method has results are used for ranks and tests (methods that are not
applicable everywhere, e.g. TabPFN with > 10 classes, are reported separately on their subset).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HIGHER_BETTER = {"acc": True, "auc": True, "logloss": False, "ece": False}


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, (len(p) - i) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def table(agg, methods, ref, metric):
    sub = agg[agg.method.isin(methods)]
    piv = sub.pivot(index="dataset", columns="method", values=metric).dropna()
    asc = not HIGHER_BETTER[metric]
    ranks = piv.rank(axis=1, ascending=asc).mean().sort_values()
    rows = []
    for m in ranks.index:
        r = {"method": m, "rank": ranks[m], "mean": piv[m].mean()}
        if "arith_mean" in piv and metric == "logloss":
            r["rel_to_avg"] = (piv[m] / piv["arith_mean"]).mean()
        if ref in piv and m != ref:
            d = piv[ref] - piv[m]
            better = d > 1e-9 if HIGHER_BETTER[metric] else d < -1e-9
            worse = d < -1e-9 if HIGHER_BETTER[metric] else d > 1e-9
            r["W/T/L(ref)"] = f"{better.sum()}/{(~better & ~worse).sum()}/{worse.sum()}"
            r["p"] = wilcoxon(piv[ref], piv[m]).pvalue if (d.abs() > 1e-12).sum() >= 5 else np.nan
        rows.append(r)
    out = pd.DataFrame(rows)
    if "p" in out:
        mask = out.p.notna()
        out.loc[mask, "p_holm"] = holm(out.loc[mask, "p"])
    return out, len(piv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--ref", default=None, help="reference method (default: first FusionPFN method)")
    ap.add_argument("--metric", nargs="*", default=["logloss", "acc", "ece"])
    args = ap.parse_args()
    df = pd.concat([pd.read_csv(c) for c in args.csv]).drop_duplicates(["dataset", "fold", "method"], keep="last")
    agg = df.groupby(["dataset", "method"])[["logloss", "acc", "ece", "auc", "sec"]].mean().reset_index()
    ref = args.ref or next((m for m in sorted(agg.method.unique()) if m.startswith("FusionPFN")), None)
    counts = agg.groupby("method").dataset.nunique()
    full = counts[counts == counts.max()].index.tolist()
    partial = counts[counts < counts.max()].index.tolist()
    pd.set_option("display.width", 200)
    for metric in args.metric:
        out, n = table(agg, full, ref, metric)
        print(f"\n=== {metric}  ({n} datasets, ref={ref})")
        print(out.round(4).to_string(index=False))
        for m in partial:
            o, n2 = table(agg, [m, ref] + (["arith_mean"] if "arith_mean" in full else []), ref, metric)
            print(f"  [{m} on its {n2}-dataset subset]")
            print("  " + o.round(4).to_string(index=False).replace("\n", "\n  "))
    print("\nmean seconds per fold:")
    print(agg.groupby("method").sec.mean().sort_values().round(2).to_string())


if __name__ == "__main__":
    main()

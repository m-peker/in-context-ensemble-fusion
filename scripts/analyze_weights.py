"""Interpretability analysis of FusionPFN's instance-wise model weights.

For every fold we obtain w_ik (query row i, base model k) and report
  1. mean weight per base-model family (are over-confident models such as NB suppressed?)
  2. per-fold Spearman correlation between a model's mean weight and its test log-loss
     (negative = better models receive more weight)
  3. instance level: AUC of w_ik for predicting whether model k is correct on row i
     (computed within each (fold, model) and averaged; > 0.5 = weights track local competence)
Usage: python scripts/analyze_weights.py --ckpt runs/ckpt/fusionpfn.pt --committees committees
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from evaluate import iter_folds  # noqa: E402
from fusionpfn.fuse import fusionpfn_predict, load_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--committees", default="committees")
    ap.add_argument("--split", default="dev")
    args = ap.parse_args()
    model = load_model(os.path.join(ROOT, args.ckpt))
    fam, corr, inst = [], [], []
    for meta, fold, path in iter_folds(args.split, os.path.join(ROOT, "runs", args.committees)):
        z = np.load(path)
        P_oof, y, P_te, yt = z["P_oof"], z["y_tr"].astype(int), z["P_te"], z["y_te"].astype(int)
        names = list(z["models"])
        _, W = fusionpfn_predict(model, P_oof, y, P_te, z["X_tr"], z["X_te"], return_weights=True)
        Pn = np.clip(P_te, 1e-6, None); Pn /= Pn.sum(-1, keepdims=True)
        ll = -np.log(Pn[np.arange(len(yt)), :, yt]).mean(0)  # (K,) test log-loss per model
        correct = P_te.argmax(-1) == yt[:, None]
        wm = W.mean(0)
        wa = np.abs(W).mean(0)  # exponents can be negative: shares are defined on |w|
        for k, nm in enumerate(names):
            fam.append({"dataset": meta["name"], "fold": fold, "model": nm, "mean_w": wm[k],
                        "share": wa[k] / max(wa.sum(), 1e-9), "neg_frac": float((W[:, k] < 0).mean()),
                        "test_ll": ll[k]})
            if 0 < correct[:, k].mean() < 1:
                inst.append({"dataset": meta["name"], "fold": fold, "model": nm,
                             "auc": roc_auc_score(correct[:, k], W[:, k])})
        corr.append({"dataset": meta["name"], "fold": fold, "spearman": spearmanr(wm, ll).correlation})
    fam, corr, inst = pd.DataFrame(fam), pd.DataFrame(corr), pd.DataFrame(inst)
    out = os.path.join(ROOT, "results", f"weights_{os.path.basename(args.ckpt)[:-3]}_{args.split}_{args.committees}")
    fam.to_csv(out + "_family.csv", index=False); inst.to_csv(out + "_instance.csv", index=False)
    print("1) mean weight share per model family (and mean test log-loss rank):")
    fam["ll_rank"] = fam.groupby(["dataset", "fold"]).test_ll.rank()
    print(fam.groupby("model")[["share", "ll_rank"]].mean().sort_values("share", ascending=False).round(3).to_string())
    print(f"\n2) Spearman(mean weight, test log-loss) per fold: mean {corr.spearman.mean():.3f}, "
          f"median {corr.spearman.median():.3f}, share < 0: {(corr.spearman < 0).mean():.2f}")
    print(f"\n3) instance-level AUC of w_ik for 'model k correct on row i': mean {inst.auc.mean():.3f}, "
          f"share > 0.5: {(inst.auc > 0.5).mean():.2f}")
    print(inst.groupby("model").auc.mean().round(3).to_string())


if __name__ == "__main__":
    main()

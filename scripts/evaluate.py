"""Evaluate fusion methods on precomputed committees.

Usage: python scripts/evaluate.py --split dev --ckpt runs/ckpt/fusionpfn.pt --out results/dev_fusionpfn.csv
Splits: dev = fixed ~25% of datasets (hash of name) used for ALL design decisions; test = the rest.
"""
from __future__ import annotations

import os as _os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    _os.environ.setdefault(_v, "2")
_os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")
try:
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if _os.name == "nt" else 10)
except Exception:
    pass

import argparse
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from bench.baselines import BASELINES  # noqa: E402

COMMITTEES = os.path.join(ROOT, "runs", "committees")


def split_of(name: str) -> str:
    return "dev" if int(hashlib.md5(name.encode()).hexdigest(), 16) % 4 == 0 else "test"


def ece(P, y, bins=15):
    conf, pred = P.max(1), P.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return e


def metrics(P, y):
    P = np.clip(P, 1e-15, None); P = P / P.sum(1, keepdims=True)
    C = P.shape[1]
    out = {"logloss": -np.log(P[np.arange(len(y)), y]).mean(), "acc": (P.argmax(1) == y).mean(),
           "ece": ece(P, y)}
    try:
        present = np.unique(y)
        if C == 2:
            out["auc"] = roc_auc_score(y, P[:, 1])
        elif len(present) == C:
            out["auc"] = roc_auc_score(y, P, multi_class="ovr", average="macro")
        else:
            out["auc"] = np.nan
    except ValueError:
        out["auc"] = np.nan
    return out


def iter_folds(split, root=COMMITTEES):
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        meta_p = os.path.join(d, "meta.json")
        if not os.path.exists(meta_p):
            continue
        meta = json.load(open(meta_p))
        if split != "all" and split_of(meta["name"]) != split:
            continue
        for f in sorted(glob.glob(os.path.join(d, "fold*.npz"))):
            yield meta, int(f[-5]), f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--ckpt", nargs="*", default=[])
    ap.add_argument("--n-perm", type=int, default=1)
    ap.add_argument("--max-ctx", type=int, default=2048)
    ap.add_argument("--tag", default="", help="suffix for the FusionPFN method name")
    ap.add_argument("--baselines", nargs="*", default=list(BASELINES))
    ap.add_argument("--max-train", type=int, default=None, help="subsample training rows (small-data regime)")
    ap.add_argument("--committees", default="committees", help="subfolder of runs/ (e.g. committees_n128)")
    ap.add_argument("--shard", default="0/1", help="i/n: process every n-th fold starting at i (parallel runs)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    shard_i, shard_n = map(int, args.shard.split("/"))

    models = {}
    if args.ckpt:
        from fusionpfn.fuse import fusionpfn_predict, load_model
        for c in args.ckpt:
            models["FusionPFN:" + os.path.basename(c)[:-3] + args.tag] = load_model(c)

    # incremental + resumable: every finished fold is appended to <out>.partial.csv
    out_path = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    partial = out_path + (f".shard{shard_i}of{shard_n}" if shard_n > 1 else "") + ".partial.csv"
    done = set()
    if os.path.exists(partial):
        prev = pd.read_csv(partial)
        done = set(zip(prev.dataset, prev.fold))
        print(f"resuming: {len(done)} folds already done", flush=True)
    for j, (meta, fold, path) in enumerate(iter_folds(args.split, os.path.join(ROOT, "runs", args.committees))):
        if j % shard_n != shard_i or (meta["name"], fold) in done:
            continue
        rows = []
        z = np.load(path)
        P_oof, y_tr, P_te, y_te = z["P_oof"], z["y_tr"].astype(int), z["P_te"], z["y_te"].astype(int)
        X_tr, X_te = z["X_tr"], z["X_te"]
        if args.max_train and len(y_tr) > args.max_train:
            rng = np.random.default_rng(fold)
            idx = rng.choice(len(y_tr), args.max_train, replace=False)
            P_oof, y_tr, X_tr = P_oof[idx], y_tr[idx], X_tr[idx]
        for name in args.baselines:
            t0 = time.time()
            P = BASELINES[name](P_oof, y_tr, P_te, X_tr=X_tr, X_te=X_te)
            if P is None:  # method not applicable (e.g. TabPFN with > 10 classes)
                continue
            rows.append({"dataset": meta["name"], "fold": fold, "method": name, "sec": time.time() - t0,
                         **metrics(P, y_te)})
        for name, m in models.items():
            t0 = time.time()
            P = fusionpfn_predict(m, P_oof, y_tr, P_te, X_tr, X_te, n_perm=args.n_perm, max_ctx=args.max_ctx)
            rows.append({"dataset": meta["name"], "fold": fold, "method": name, "sec": time.time() - t0,
                         **metrics(P, y_te)})
        pd.DataFrame(rows).to_csv(partial, mode="a", header=not os.path.exists(partial), index=False)
        print(f"{meta['name']} f{fold} n_tr={len(y_tr)}", flush=True)
    if shard_n > 1:
        print("shard finished; merge shards with scripts/report.py (it accepts several csv files)")
        return
    df = pd.read_csv(partial)
    df.to_csv(out_path, index=False)
    os.remove(partial)
    summarize(df)


def summarize(df):
    agg = df.groupby(["dataset", "method"])[["logloss", "acc", "ece", "auc"]].mean().reset_index()
    ranks = agg.copy()
    for m, asc in (("logloss", True), ("acc", False), ("ece", True), ("auc", False)):
        ranks[m] = agg.groupby("dataset")[m].rank(ascending=asc)
    print(f"\n{agg.dataset.nunique()} datasets — mean rank (lower is better)")
    print(ranks.groupby("method")[["logloss", "acc", "ece", "auc"]].mean().sort_values("logloss").round(2))
    # normalized log-loss relative to arith_mean per dataset
    if "arith_mean" in set(agg.method):
        ref = agg[agg.method == "arith_mean"].set_index("dataset")["logloss"]
        agg["ll_rel"] = agg.logloss / agg.dataset.map(ref)
        print("\nmean log-loss relative to arith_mean (<1 better)")
        print(agg.groupby("method")["ll_rel"].mean().sort_values().round(4))


if __name__ == "__main__":
    main()

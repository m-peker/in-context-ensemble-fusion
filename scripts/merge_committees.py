"""Merge the original (8 learners) and the unseen-learner (6 learners) committees into K=14 committees.

Both were built with identical splits and seeds, so rows align; this is verified for every fold.
Usage: python scripts/merge_committees.py   (all regimes for which committees_alt* exist)
"""
import glob
import json
import os
import shutil
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from bench.committee import save_split  # noqa: E402

for suffix in ["", "_n128", "_n512"]:
    alt_root = os.path.join(ROOT, "runs", f"committees_alt{suffix}")
    if not os.path.isdir(alt_root):
        continue
    n = 0
    for alt in sorted(glob.glob(os.path.join(alt_root, "*", "fold*.npz"))):
        ds, fold = alt.split(os.sep)[-2], os.path.basename(alt)
        orig = os.path.join(ROOT, "runs", f"committees{suffix}", ds, fold)
        out = os.path.join(ROOT, "runs", f"committees_union{suffix}", ds, fold)
        if os.path.exists(out) or not os.path.exists(orig):
            continue
        a, o = np.load(alt), np.load(orig)
        assert (a["y_tr"] == o["y_tr"]).all() and (a["y_te"] == o["y_te"]).all(), (ds, fold)
        save_split(out, P_oof=np.concatenate([o["P_oof"], a["P_oof"]], 1), P_te=np.concatenate([o["P_te"], a["P_te"]], 1),
                   y_tr=o["y_tr"], y_te=o["y_te"], X_tr=o["X_tr"], X_te=o["X_te"],
                   models=np.concatenate([o["models"], a["models"]]))
        meta = os.path.join(alt_root, ds, "meta.json")
        shutil.copy(meta, os.path.join(os.path.dirname(out), "meta.json"))
        n += 1
    print(f"committees_union{suffix}: {n} folds merged")

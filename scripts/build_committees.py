"""Precompute base-model committees (OOF + test probabilities) on OpenML-CC18.

Resumable: existing fold files are skipped.
Usage: python scripts/build_committees.py [--limit N] [--jobs J] [--max-rows 10000]
"""
from __future__ import annotations

import os as _os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
_os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np
import openml
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from bench.committee import ALT_MODELS, BASE_MODELS, build_committee, make_preprocessor, save_split  # noqa: E402

OUT = os.path.join(ROOT, "runs", "committees")
CACHE = os.path.join(ROOT, "data", "openml_cache")
SEED = 42
OUTER_FOLDS = 5
openml.config.set_root_cache_directory(CACHE)  # module level: spawned workers need it too


def list_cc18(max_features: int):
    suite = openml.study.get_suite(99)
    df = openml.tasks.list_tasks(task_id=suite.tasks, output_format="dataframe")
    df = df[df["NumberOfFeatures"] <= max_features + 1]
    df = df.sort_values("NumberOfInstances")
    return df[["tid", "did", "name", "NumberOfInstances", "NumberOfFeatures", "NumberOfClasses"]]


def load_dataset(tid: int, max_rows: int):
    task = openml.tasks.get_task(tid, download_splits=False)
    ds = task.get_dataset()
    X, y, cat_mask, _ = ds.get_data(target=task.target_name, dataset_format="dataframe")
    y = y.astype(str).to_numpy()
    classes, y = np.unique(y, return_inverse=True)
    # drop very rare classes (cannot be stratified)
    counts = np.bincount(y)
    keep_cls = np.where(counts >= 10)[0]
    mask = np.isin(y, keep_cls)
    X, y = X[mask].reset_index(drop=True), y[mask]
    y = np.searchsorted(keep_cls, y)
    if len(y) > max_rows:
        idx, _ = train_test_split(np.arange(len(y)), train_size=max_rows, stratify=y, random_state=SEED)
        idx = np.sort(idx)
        X, y = X.iloc[idx].reset_index(drop=True), y[idx]
    for c in X.columns[np.array(cat_mask)]:
        X[c] = X[c].astype(str)
    return X, y, np.array(cat_mask), ds.name


def run_fold(tid, did, max_rows, fold, n_train=None, alt=False):
    """n_train: true small-data regime -- base models see only n_train training rows."""
    t0 = time.time()
    X, y, cat_mask, name = load_dataset(tid, max_rows)
    out_root = (OUT + "_alt" if alt else OUT) if n_train is None else (OUT + ("_alt" if alt else "") + f"_n{n_train}")
    out_dir = os.path.join(out_root, f"{did}_{name}")
    path = os.path.join(out_dir, f"fold{fold}.npz")
    if os.path.exists(path):
        return f"skip {name} f{fold}"
    skf = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED)
    tr, te = list(skf.split(X, y))[fold]
    if n_train is not None:
        rs = np.random.RandomState(SEED + fold)
        if len(tr) > n_train:
            tr = _strat_subsample(tr, y, n_train, rs)
        if len(te) > MAX_TEST_SMALL:
            te = _strat_subsample(te, y, MAX_TEST_SMALL, rs)
    pre = make_preprocessor(cat_mask)
    Xtr = pre.fit_transform(X.iloc[tr]).astype(np.float32)
    Xte = pre.transform(X.iloc[te]).astype(np.float32)
    C = int(y.max()) + 1
    models = ALT_MODELS if alt else BASE_MODELS
    P_oof, P_te = build_committee(Xtr, y[tr], Xte, C, seed=SEED, models=models)
    save_split(path, P_oof=P_oof, P_te=P_te, y_tr=y[tr].astype(np.int16), y_te=y[te].astype(np.int16),
               X_tr=Xtr, X_te=Xte, models=np.array(models))
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"tid": int(tid), "did": int(did), "name": name, "n": int(len(y)), "d": int(X.shape[1]),
                   "C": C}, f)
    return f"done {name} f{fold} n={len(y)} C={C} {time.time() - t0:.0f}s"


MAX_TEST_SMALL = 2000


def _strat_subsample(idx, y, n, rs):
    """Stratified subsample of idx (keeps >= 2 rows of every class when possible)."""
    yy = y[idx]
    classes, counts = np.unique(yy, return_counts=True)
    quota = np.maximum(np.minimum(counts, 2), np.floor(counts / counts.sum() * n)).astype(int)
    while quota.sum() > n:
        quota[np.argmax(quota)] -= 1
    rest = n - quota.sum()
    pick = [rs.choice(idx[yy == c], q, replace=False) for c, q in zip(classes, quota)]
    chosen = np.concatenate(pick)
    if rest > 0:
        remaining = np.setdiff1d(idx, chosen)
        chosen = np.concatenate([chosen, rs.choice(remaining, rest, replace=False)])
    return np.sort(chosen)


def safe_run(*a):
    try:
        return run_fold(*a)
    except Exception:
        return f"FAIL {a}: {traceback.format_exc(limit=3)}"


def lower_priority():
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except Exception:
        pass


def main():
    lower_priority()  # child workers inherit the priority class
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--max-rows", type=int, default=10000)
    ap.add_argument("--max-features", type=int, default=500)
    ap.add_argument("--n-train", type=int, default=None, help="true small-data regime")
    ap.add_argument("--alt", action="store_true", help="use ALT_MODELS (learners unseen during pretraining)")
    ap.add_argument("--only-test", action="store_true", help="only datasets of the TEST split")
    args = ap.parse_args()
    openml.config.set_root_cache_directory(CACHE)
    tasks = list_cc18(args.max_features)
    if args.limit:
        tasks = tasks.head(args.limit)
    print(f"{len(tasks)} tasks", flush=True)
    # download sequentially first (avoid concurrent cache writes)
    for r in tasks.itertuples():
        try:
            load_dataset(r.tid, args.max_rows)
        except Exception as e:
            print("download fail", r.name, e, flush=True)
    if args.only_test:
        import hashlib
        is_test = lambda nm: int(hashlib.md5(nm.encode()).hexdigest(), 16) % 4 != 0
        names = {r.tid: openml.tasks.get_task(r.tid, download_splits=False).get_dataset().name for r in tasks.itertuples()}
        tasks = tasks[[is_test(names[t]) for t in tasks.tid]]
        print(f"{len(tasks)} test tasks", flush=True)
    jobs = [(r.tid, r.did, args.max_rows, f, args.n_train, args.alt) for r in tasks.itertuples() for f in range(OUTER_FOLDS)]
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(safe_run, *j) for j in jobs]
        for fu in as_completed(futs):
            print(fu.result(), flush=True)


if __name__ == "__main__":
    main()

"""Batch sampler over the simulated-committee pool (runs/sim_pool)."""
from __future__ import annotations

import glob
import os
import random
from collections import defaultdict

import numpy as np
import torch


class SimPool:
    def __init__(self, root, max_tasks=None, seed=0):
        files = sorted(glob.glob(os.path.join(root, "task_*.npz")))
        files = [f for f in files if not f.endswith(".tmp.npz")]
        if max_tasks:
            files = files[:max_tasks]
        self.rng = random.Random(seed)
        self.by_C = defaultdict(list)
        for f in files:
            z = np.load(f)
            t = {k: z[k] for k in ("P_oof", "P_te", "y_tr", "y_te", "X_tr", "X_te")}
            self.by_C[t["P_oof"].shape[-1]].append(t)
        self.n = sum(len(v) for v in self.by_C.values())
        print(f"SimPool: {self.n} tasks, classes: { {c: len(v) for c, v in sorted(self.by_C.items())} }", flush=True)

    def sample(self, B, n_ctx, n_qry, device="cuda"):
        """Return a batch dict compatible with prior.sample_committee_tasks (K, C chosen by the pool)."""
        Cs = [c for c, v in self.by_C.items() if len(v) >= 1]
        C = self.rng.choice(Cs)
        tasks = [self.rng.choice(self.by_C[C]) for _ in range(B)]
        K = min(t["P_oof"].shape[1] for t in tasks)
        K = self.rng.randint(min(2, K), K)
        n_ctx = min(n_ctx, *(len(t["y_tr"]) for t in tasks))
        n_qry = min(n_qry, *(len(t["y_te"]) for t in tasks))
        d = max(t["X_tr"].shape[1] for t in tasks)
        out = {k: [] for k in ("P_ctx", "y_ctx", "X_ctx", "P_qry", "y_qry", "X_qry")}
        for t in tasks:
            models = self.rng.sample(range(t["P_oof"].shape[1]), K)
            rc = self.rng.sample(range(len(t["y_tr"])), n_ctx)
            rq = self.rng.sample(range(len(t["y_te"])), n_qry)
            pad = lambda X: np.pad(X.astype(np.float32), ((0, 0), (0, d - X.shape[1])))  # zero-pad keeps distances
            out["P_ctx"].append(t["P_oof"][rc][:, models].astype(np.float32))
            out["y_ctx"].append(t["y_tr"][rc].astype(np.int64))
            out["X_ctx"].append(pad(t["X_tr"][rc]))
            out["P_qry"].append(t["P_te"][rq][:, models].astype(np.float32))
            out["y_qry"].append(t["y_te"][rq].astype(np.int64))
            out["X_qry"].append(pad(t["X_te"][rq]))
        res = {k: torch.as_tensor(np.stack(v), device=device) for k, v in out.items()}
        for k in ("P_ctx", "P_qry"):  # float16 storage can break normalisation slightly
            res[k] = res[k].clamp_min(0)
            res[k] = res[k] / res[k].sum(-1, keepdim=True).clamp_min(1e-6)
        return res, K, C

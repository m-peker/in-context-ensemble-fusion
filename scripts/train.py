"""Pretrain FusionPFN on the synthetic committee prior.

Usage: python scripts/train.py --name fusionpfn --steps 20000
The full multi-stage recipe of the released model is scripts/pretrain.sh.
Resumable from runs/ckpt/<name>.pt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from fusionpfn.features import cell_features, class_features  # noqa: E402
from fusionpfn.model import FusionPFN  # noqa: E402
from fusionpfn.prior import sample_committee_tasks  # noqa: E402


def sample_shape(rng, tokens=120_000):
    K = rng.randint(2, 12)
    C = rng.choice([2, 2, 2, 3, 3, 4, 5, 6, 8, 10])
    n_ctx = int(math.exp(rng.uniform(math.log(32), math.log(1024))))
    n_qry = 256
    B = max(2, min(32, int(tokens / ((n_ctx + n_qry) * K))))
    # row-axis attention materializes (B*K, heads, N, n_ctx) on Windows (no flash kernel): cap it
    B = max(1, min(B, int(4e7 / (K * (n_ctx + n_qry) * n_ctx))))
    return B, n_ctx, n_qry, K, C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="fusionpfn")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-loc", action="store_true", help="ablation: drop locality (kNN) features")
    ap.add_argument("--residual", action="store_true", help="add the class-equivariant correction head")
    ap.add_argument("--init", default=None, help="warm-start from checkpoint (non-strict)")
    ap.add_argument("--sim-pool", default=None, help="dir of simulated committees (real models, synthetic data)")
    ap.add_argument("--sim-frac", type=float, default=0.5, help="share of batches drawn from the sim pool")
    ap.add_argument("--reload-every", type=int, default=0, help="re-scan the sim pool every N steps (pool still growing)")
    ap.add_argument("--p-confuse", type=float, default=None, help="override PRIOR['p_confuse'] (probability of systematic confusions)")
    ap.add_argument("--tokens", type=int, default=120_000, help="cells per batch (memory budget)")
    ap.add_argument("--mem-frac", type=float, default=0.75, help="cap on the share of GPU memory this process may use")
    args = ap.parse_args()

    if args.p_confuse is not None:
        from fusionpfn.prior import PRIOR
        PRIOR["p_confuse"] = args.p_confuse
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    dev = "cuda"
    torch.cuda.set_per_process_memory_fraction(args.mem_frac)
    model = FusionPFN(d=args.d, layers=args.layers, heads=args.heads, residual=args.residual).to(dev)
    if args.init:
        missing, unexpected = model.load_state_dict(torch.load(args.init, map_location=dev)["model"], strict=False)
        print(f"warm start from {args.init}; new params: {len(missing)}, unexpected: {len(unexpected)}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warm = 1000
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    os.makedirs(os.path.join(ROOT, "runs", "ckpt"), exist_ok=True)
    ckpt = os.path.join(ROOT, "runs", "ckpt", f"{args.name}.pt")
    step = 0
    if os.path.exists(ckpt):
        st = torch.load(ckpt, map_location=dev)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        step = st["step"]; rng.setstate(st["rng"])
        print(f"resumed at step {step}", flush=True)
    with open(os.path.join(ROOT, "runs", "ckpt", f"{args.name}.json"), "w") as f:
        json.dump(vars(args), f)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M", flush=True)

    pool = None
    if args.sim_pool:
        from fusionpfn.pool import SimPool
        pool_dir = os.path.join(ROOT, args.sim_pool) if not os.path.isabs(args.sim_pool) else args.sim_pool
        pool = SimPool(pool_dir, seed=args.seed)
    log = {"loss": [], "geo": [], "ari": [], "sim": []}
    t0 = time.time()
    while step < args.steps:
        if pool is not None and args.reload_every and step > 0 and step % args.reload_every == 0:
            pool = SimPool(pool_dir, seed=args.seed + step)
        B, n_ctx, n_qry, K, C = sample_shape(rng, args.tokens)
        from_pool = pool is not None and rng.random() < args.sim_frac
        if from_pool:  # budget computed for the largest committee to stay within memory
            Bp = max(1, min(32, int(args.tokens / ((n_ctx + n_qry) * 12)), int(4e7 / (12 * (n_ctx + n_qry) * n_ctx))))
            d, K, C = pool.sample(Bp, n_ctx, n_qry, device=dev)
        else:
            d = sample_committee_tasks(B, n_ctx, n_qry, K, C, device=dev)
        Xc, Xq = (None, None) if args.no_loc else (d["X_ctx"], d["X_qry"])
        F_ctx, F_qry, _, logP_qry = cell_features(d["P_ctx"], d["y_ctx"], d["P_qry"], Xc, Xq)
        G = class_features(d["P_ctx"], d["y_ctx"], d["P_qry"]) if args.residual else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logp, _ = model(F_ctx, F_qry, logP_qry, G_qry=G)
        loss = torch.nn.functional.nll_loss(logp.float().reshape(-1, C), d["y_qry"].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); step += 1
        with torch.no_grad():
            y = d["y_qry"].reshape(-1)
            geo = torch.log_softmax(logP_qry.mean(2), -1).reshape(-1, C)
            ari = d["P_qry"].mean(2).clamp_min(1e-6).log().reshape(-1, C)
            log["loss"].append(loss.item())
            if from_pool:
                log["sim"].append(loss.item() - torch.nn.functional.nll_loss(geo, y).item())
            log["geo"].append(torch.nn.functional.nll_loss(geo, y).item())
            log["ari"].append(torch.nn.functional.nll_loss(ari, y).item())
        if step % 200 == 0:
            m = {k: float(np.mean(v)) if v else float("nan") for k, v in log.items()}
            print(f"step {step} loss {m['loss']:.4f} geo-mean {m['geo']:.4f} arith-mean {m['ari']:.4f} "
                  f"sim(loss-geo) {m['sim']:.4f} "
                  f"lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True)
            log = {k: [] for k in log}
        if step % 1000 == 0 or step == args.steps:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "step": step, "rng": rng.getstate(), "args": vars(args)}, ckpt + ".tmp")
            os.replace(ckpt + ".tmp", ckpt)


if __name__ == "__main__":
    main()

"""Build the simulated-committee pool (real sklearn models on synthetic datasets).

Resumable; one file per task: runs/sim_pool/task_<seed>.npz (float16 storage).
Usage: python scripts/build_sim_pool.py --n 8000 --jobs 6
"""
import os as _os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
_os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import argparse  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
OUT = os.path.join(ROOT, "runs", "sim_pool")
SEED_OFFSET = 10_000_000  # keep pool seeds disjoint from anything else


def work(i):
    from fusionpfn.sim_committees import simulate_task
    path = os.path.join(OUT, f"task_{i:06d}.npz")
    if os.path.exists(path) or os.path.exists(path + ".none"):
        return "skip"
    t0 = time.time()
    try:
        r = simulate_task(SEED_OFFSET + i)
    except Exception as e:  # degenerate synthetic task
        r = None
        err = repr(e)[:80]
    if r is None:
        open(path + ".none", "w").close()
        return "none"
    for k in ("P_oof", "P_te", "X_tr", "X_te"):
        r[k] = r[k].astype(np.float16)
    tmp = path[:-4] + ".tmp.npz"
    np.savez_compressed(tmp, **r)
    os.replace(tmp, path)
    return f"ok {time.time() - t0:.0f}s"


def main():
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0, done = time.time(), 0
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(work, i) for i in range(args.n)]
        for fu in as_completed(futs):
            done += 1
            msg = fu.result()
            if done % 100 == 0 or msg.startswith("ERR"):
                print(f"{done}/{args.n} {time.time() - t0:.0f}s last={msg}", flush=True)


if __name__ == "__main__":
    main()

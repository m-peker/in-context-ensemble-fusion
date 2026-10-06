"""Figures and a LaTeX table from evaluation CSVs (scripts/evaluate.py output).

  results/figures/learning_curve.pdf   relative log-loss vs. training regime (95% bootstrap CI over data sets)
  results/figures/cd_<regime>.pdf      critical-difference diagrams (Friedman + Nemenyi, Demsar 2006) on log-loss
  results/tables/main_results.tex      relative log-loss, mean rank, W/L and Holm-corrected Wilcoxon tests
Usage: python scripts/plots.py --prefix test --ref FusionPFN:fusionpfn_p4
       (reads results/<prefix>_*_<committees>.csv for the three regimes)
"""
from __future__ import annotations

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import friedmanchisquare, studentized_range, wilcoxon  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, FIG, TAB = (os.path.join(ROOT, p) for p in ("results", "results/figures", "results/tables"))
PREFIX = "test"
REF = "FusionPFN:fusionpfn_p4"  # overridden by --ref
REGIMES = [("committees_n128", "128"), ("committees_n512", "512"), ("committees", "full")]
NAMES = {REF: "FusionPFN", "NE_stack": "NE (stack)", "NE_avg": "NE (avg)", "TabPFN_stack": "TabPFN-stacker",
         "greedy_ES": "Greedy ES", "stack_LR": "Stacking (LR)", "single_best": "Single best",
         "arith_mean": "Average", "geo_mean": "Geometric mean", "majority_vote": "Majority vote", "KNORA-E": "KNORA-E"}
# validated categorical palette (dataviz reference, light mode), fixed order -> entity
STYLE = {REF: ("#2a78d6", "o"), "NE_stack": ("#eb6834", "s"), "TabPFN_stack": ("#1baf7a", "^"),
         "greedy_ES": ("#eda100", "D"), "stack_LR": ("#e87ba4", "v"), "NE_avg": ("#4a3aa7", "P")}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})


def load(regime):
    fs = glob.glob(os.path.join(RES, f"{PREFIX}_*_{regime}.csv"))
    if not fs:
        return None
    df = pd.concat([pd.read_csv(f) for f in fs]).drop_duplicates(["dataset", "fold", "method"], keep="last")
    return df.groupby(["dataset", "method"])[["logloss", "acc", "ece", "sec"]].mean().reset_index()


def rel_logloss(agg):
    piv = agg.pivot(index="dataset", columns="method", values="logloss")
    return piv.div(piv["arith_mean"], axis=0)


def boot_ci(x, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    m = rng.choice(x, (B, len(x)), replace=True).mean(1)
    return x.mean(), *np.percentile(m, [2.5, 97.5])


def learning_curve(data):
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    xs = np.arange(len(REGIMES))
    ends = []
    for mi, (meth, (col, mk)) in enumerate(STYLE.items()):
        dodge = (mi - (len(STYLE) - 1) / 2) * 0.035  # separate overlapping CI bars
        pts = []
        for j, (reg, _) in enumerate(REGIMES):
            if data.get(reg) is None or meth not in data[reg]:
                continue
            v = data[reg][meth].dropna()
            pts.append((j + dodge, *boot_ci(v)))
        if not pts:
            continue
        p = np.array(pts)
        ax.plot(p[:, 0], p[:, 1], color=col, lw=2, marker=mk, ms=6, mec="white", mew=1, zorder=3)
        ax.vlines(p[:, 0], p[:, 2], p[:, 3], color=col, lw=1.2, alpha=0.6, zorder=2)
        ends.append((p[-1, 0], p[-1, 1], NAMES[meth]))
    # direct labels at line ends, nudged apart so they never collide
    lo_y, hi_y = ax.get_ylim()
    gap = 0.045 * (hi_y - lo_y)
    for x0 in sorted({round(e[0]) for e in ends}):
        grp = sorted([e for e in ends if round(e[0]) == x0], key=lambda e: e[1])
        ys = [e[1] for e in grp]
        for t in range(1, len(ys)):
            ys[t] = max(ys[t], ys[t - 1] + gap)
        for (x, y, name), yl in zip(grp, ys):
            ax.annotate(name, (x, y), xytext=(x0 + 0.24, yl), textcoords="data", va="center", fontsize=7.5,
                        color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkA=0, shrinkB=3))
    ax.set_xticks(xs, [f"n = {r}" if r != "full" else "full data" for _, r in REGIMES])
    ax.set_xlim(-0.3, len(REGIMES) + 0.05)
    ax.set_ylabel("log-loss relative to averaging (lower is better)")
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "learning_curve.pdf"))
    plt.close(fig)


def cd_diagram(piv, title, path, alpha=0.05):
    piv = piv.dropna()
    ranks = piv.rank(axis=1).mean().sort_values()
    k, N = len(ranks), len(piv)
    cd = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * N))
    p_f = friedmanchisquare(*[piv[c] for c in piv.columns]).pvalue
    half = (k + 1) // 2
    rows = max(half, k - half)
    fig, ax = plt.subplots(figsize=(6.4, 1.15 + 0.22 * rows))
    lo, hi = 1, k
    ax.set_xlim(lo - 3.2, hi + 3.2)
    y_axis, step = 0.0, 1.0
    ax.set_ylim(-(rows + 1.2) * step, 1.6)
    ax.axis("off")
    ax.hlines(y_axis, lo, hi, color=INK, lw=0.8)
    for t in range(lo, hi + 1):
        ax.vlines(t, y_axis, y_axis + 0.18, color=INK, lw=0.8)
        ax.text(t, y_axis + 0.3, str(t), ha="center", va="bottom", fontsize=7, color=MUTED)
    ax.hlines(1.25, lo, lo + cd, color=INK, lw=1.4)
    ax.vlines([lo, lo + cd], 1.15, 1.35, color=INK, lw=1.0)
    ax.text(lo + cd + 0.1, 1.25, f"CD = {cd:.2f}", ha="left", va="center", fontsize=7, color=INK)
    vals = ranks.values
    # cliques (maximal groups with rank range < CD) drawn just below the axis
    cliques = []
    for i in range(k):
        j = i
        while j + 1 < k and vals[j + 1] - vals[i] < cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in cliques):
            cliques.append((i, j))
    for c, (i, j) in enumerate(cliques):
        y = -0.35 - 0.22 * (c % 3)
        ax.hlines(y, vals[i] - 0.04, vals[j] + 0.04, color=INK, lw=2.0)
    for i, m in enumerate(ranks.index):
        r = ranks[m]
        left = i < half
        row = i if left else (k - 1 - i)
        y = -(row + 1.4) * step
        xt = lo - 0.4 if left else hi + 0.4
        col = STYLE[m][0] if m in STYLE else MUTED
        ax.plot([r, r, xt], [y_axis, y, y], color=col, lw=1)
        ax.text(xt + (-0.1 if left else 0.1), y, f"{NAMES.get(m, m)} ({r:.2f})", ha="right" if left else "left",
                va="center", fontsize=7.5, color=INK, fontweight="bold" if m == REF else "normal")
    fig.suptitle(f"{title}  (N = {N} data sets, Friedman p = {p_f:.1e})", fontsize=8, color=INK, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); out = np.empty_like(p); run = 0
    for i, idx in enumerate(o):
        run = max(run, (len(p) - i) * p[idx]); out[idx] = min(1, run)
    return out


def main_table(data, raw):
    order = [REF, "NE_stack", "TabPFN_stack", "greedy_ES", "stack_LR", "NE_avg", "single_best", "arith_mean",
             "geo_mean", "majority_vote", "KNORA-E"]
    lines = [r"\begin{tabular}{l" + "rrr" * len(REGIMES) + "}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{'$n=' + r + '$' if r != 'full' else 'full data'}}}"
                                for _, r in REGIMES) + r" \\",
             " ".join(rf"\cmidrule(lr){{{2 + 3 * j}-{4 + 3 * j}}}" for j in range(len(REGIMES))),
             "Method & " + " & ".join(["rel.\\ LL & rank & W/L"] * len(REGIMES)) + r" \\", r"\midrule"]
    cells = {m: [] for m in order}
    for reg, _ in REGIMES:
        piv = data.get(reg)
        if piv is None:
            for m in order: cells[m] += ["--"] * 3
            continue
        full = piv.dropna(axis=1, how="any")  # methods available on every data set
        ranks = full.rank(axis=1).mean()
        ps, keys = [], []
        for m in order:
            if m == REF or m not in piv:
                continue
            d = raw[reg][[REF, m]].dropna()  # paired test on raw log-loss (as scripts/report.py)
            ps.append(wilcoxon(d[REF], d[m]).pvalue); keys.append(m)
        padj = dict(zip(keys, holm(ps)))
        best = min(piv[m].mean() for m in order if m in piv)
        for m in order:
            if m not in piv:
                cells[m] += ["--"] * 3; continue
            v = piv[m].dropna()
            s = f"{v.mean():.3f}"
            s = rf"\textbf{{{s}}}" if abs(v.mean() - best) < 1e-12 else s
            rk = f"{ranks[m]:.2f}" if m in ranks else "--"
            if m == REF:
                wl = "--"
            else:
                d = piv[[REF, m]].dropna()
                w, l = int((d[REF] < d[m]).sum()), int((d[REF] > d[m]).sum())
                star = "$^{*}$" if padj[m] < 0.05 else ""
                wl = f"{w}/{l}{star}"
            cells[m] += [s + ("$^\\dagger$" if len(v) < len(piv) else ""), rk, wl]
    for m in order:
        lines.append(f"{NAMES[m]} & " + " & ".join(cells[m]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(TAB, "main_results.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    global PREFIX, REF
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="test", help="CSV prefix, e.g. test or dev")
    ap.add_argument("--ref", default=REF, help="method name of the FusionPFN run to highlight")
    args = ap.parse_args()
    PREFIX = args.prefix
    if args.ref != REF:
        NAMES[args.ref] = NAMES.pop(REF); STYLE[args.ref] = STYLE.pop(REF)
        REF = args.ref
    os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
    data, raw = {}, {}
    for reg, lab in REGIMES:
        agg = load(reg)
        data[reg] = None if agg is None else rel_logloss(agg)
        raw[reg] = None if agg is None else agg.pivot(index="dataset", columns="method", values="logloss")
    learning_curve(data)
    for reg, lab in REGIMES:
        piv = data[reg]
        if piv is None:
            continue
        cd_diagram(piv.drop(columns=[c for c in piv.columns if c not in NAMES]),
                   f"log-loss, {'n = ' + lab if lab != 'full' else 'full data'}", os.path.join(FIG, f"cd_{lab}.pdf"))
    main_table(data, raw)
    print("figures:", sorted(os.listdir(FIG)), "tables:", sorted(os.listdir(TAB)))


if __name__ == "__main__":
    main()

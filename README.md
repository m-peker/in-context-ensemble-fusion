# FusionPFN

**Pretrained, in-context fusion of model committees.**

Stacking, ensemble selection and dynamic ensemble selection learn a fusion rule from scratch on every data set.
FusionPFN instead *infers* the fusion rule: a transformer is pretrained once on a synthetic **prior over model
committees** and then combines the predictions of an arbitrary committee on a new data set in a single forward
pass — no meta-learner is fitted, no hyper-parameters are tuned.

Given the out-of-fold class probabilities of *K* base models on the training rows (and the labels), FusionPFN returns
fused class probabilities for new rows together with **instance-wise member weights**.

- **No fitting** on the target data set: one forward pass on CPU or GPU.
- **Any committee**: any number of members and classes, any learners — the network is equivariant to permutations
  of members, classes and rows.
- **Interpretable**: the output is a logarithmic opinion pool whose exponents (member weights) are predicted per row.

---

## Installation

Python 3.10 or 3.11 is recommended.

```bash
git clone https://github.com/m-peker/in-context-ensemble-fusion.git
cd in-context-ensemble-fusion
pip install -r requirements.txt
```

Install a PyTorch build that matches your CUDA version first if you want GPU support
(see [pytorch.org](https://pytorch.org/get-started/locally/)). Inference also runs on CPU; pretraining requires a CUDA GPU.

Pretrained weights are not distributed with this repository. Create them with the pretraining recipe
(see [Pretraining](#2-pretraining)); the examples below assume the resulting checkpoint is available as `fusionpfn.pt`
(e.g. a copy of `runs/ckpt/fusionpfn_seed42.pt`).

## Quick start

```python
import numpy as np
from sklearn.model_selection import cross_val_predict
from fusionpfn import fuse, load_model          # add "src" to PYTHONPATH

# committee: any scikit-learn compatible classifiers with predict_proba
members = [...]

# context: out-of-fold probabilities on the training rows, shape (n, K, C)
P_oof = np.stack([cross_val_predict(m, X_tr, y_tr, cv=5, method="predict_proba") for m in members], axis=1)
# queries: probabilities of the members refitted on all training rows, shape (m, K, C)
P_te = np.stack([m.fit(X_tr, y_tr).predict_proba(X_te) for m in members], axis=1)

model = load_model("fusionpfn.pt")
P, W = fuse(model, P_oof, y_tr, P_te, X_tr, X_te, n_perm=4, return_weights=True)
# P: (m, C) fused probabilities, W: (m, K) instance-wise member weights
```

A complete, runnable example is [`examples/quickstart.py`](examples/quickstart.py):

```bash
python examples/quickstart.py fusionpfn.pt
```

Labels must be integers `0 … C-1`. `X_tr`/`X_te` (standardised inputs) are optional; they feed nearest-neighbour
features and can be omitted with `X_tr=None, X_te=None`. Large contexts are sub-sampled to 2,048 rows; `n_perm`
averages over several sub-samples and member orders.

## How it works

1. **Prior over committees.** Training tasks are sampled from two generators:
   *parametric committees* — members are noisy, possibly miscalibrated readings of a latent class-logit function, with
   smooth competence maps, errors shared within latent model families, systematic class confusions, duplicated and
   uninformative members; and *simulated committees* — real learning algorithms (twelve scikit-learn families with
   random hyper-parameters) fitted on synthetic data sets, with cross-fitted out-of-fold predictions.
2. **Cell representation.** Every (row, member) cell is described by class-agnostic features (entropy, margin,
   agreement with and divergence from the committee consensus, the member's track record on the context, and
   optional nearest-neighbour statistics), plus class-conditional statistics for the output head.
3. **Two-axis transformer.** Member-axis attention lets the members of a row compare their predictions; row-axis
   attention lets each cell read the context track record of its member. Query rows never attend to each other.
4. **Output.** A logarithmic opinion pool `log p(c) ∝ Σ_k w_ik log p_k(c) + r_ic` with instance-wise exponents `w_ik`
   and a small class-equivariant correction `r_ic`. An untrained network returns the geometric mean.

## Reproducing the full pipeline

All commands are run from the repository root. Outputs go to `data/`, `runs/` and `results/` (git-ignored).

### 1. Committees on OpenML-CC18

```bash
python scripts/build_committees.py --jobs 6                    # full-data regime
python scripts/build_committees.py --jobs 6 --n-train 128      # small-data regimes: the base models
python scripts/build_committees.py --jobs 6 --n-train 512      # see only n training rows
```

This downloads the CC18 tasks (≤ 500 features), runs 5-fold outer cross-validation, and stores out-of-fold and test
probabilities of eight base models for every fold in `runs/committees*/`. Optional: `--alt --only-test` builds
committees of six further learners (LightGBM, SGD, Bernoulli NB, calibrated linear SVM / ridge, bagged trees);
`scripts/merge_committees.py` merges them with the original committees into 14-member committees.

### 2. Pretraining

```bash
python scripts/build_sim_pool.py --n 10000 --jobs 6            # simulated committees (CPU, several hours)
bash scripts/pretrain.sh 42                                     # five-stage curriculum (GPU, a few hours)
```

`scripts/pretrain.sh` documents the five stages; every stage resumes from its checkpoint.
`scripts/ablations.sh` trains and evaluates the prior and component ablations.

### 3. Evaluation and baselines

```bash
bash scripts/setup_neural_ensemblers.sh                        # official Neural Ensemblers code (baseline)
for c in committees_n128 committees_n512 committees; do
  python scripts/evaluate.py --split test --committees $c --out results/test_baselines_$c.csv
  python scripts/evaluate.py --split test --committees $c --baselines \
         --ckpt runs/ckpt/fusionpfn_seed42.pt --n-perm 4 --tag _p4 --out results/test_fusionpfn_$c.csv
done
python scripts/report.py results/test_*_committees.csv --ref FusionPFN:fusionpfn_seed42_p4
python scripts/plots.py --prefix test --ref FusionPFN:fusionpfn_seed42_p4
```

Data sets are split once into a development and a test part (`split_of` in `scripts/evaluate.py`); all design
decisions should be taken on `--split dev`. Baselines: single best, arithmetic and geometric mean, majority vote,
greedy ensemble selection, stacking with logistic regression, KNORA-E, Neural Ensemblers (stacking and averaging) and
TabPFN v2 as a stacker. Evaluation runs are incremental and can be resumed or split with `--shard i/n`.

### 4. Analysis

```bash
python scripts/analyze_weights.py --ckpt runs/ckpt/fusionpfn_seed42.pt     # what do the weights reveal?
python scripts/prior_stats.py                                              # prior vs. real committee statistics
python scripts/diag_synthetic.py fusionpfn_seed42                          # FusionPFN vs. baselines on prior tasks
```

## Repository structure

```
src/fusionpfn/      prior.py            parametric committee prior
                    sim_committees.py   simulated committees (real learners on synthetic data)
                    pool.py             batch sampler over the simulated-committee pool
                    features.py         class-agnostic cell features and class-conditional statistics
                    model.py            two-axis transformer and opinion-pool head
                    fuse.py             inference: load_model, fuse
src/bench/          committee.py        base models and out-of-fold committee construction
                    baselines.py        fusion baselines
scripts/            data preparation, pretraining, evaluation, reporting, analysis
examples/           quickstart.py
tests/              symmetry tests (member / class / row permutations, query independence)
```

## Tests

```bash
python -m pytest tests
```

## Citation

Citation information will be added.

## License

MIT — see [LICENSE](LICENSE). Baseline code that is fetched from other repositories keeps its own license.

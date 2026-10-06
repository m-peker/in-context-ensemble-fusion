#!/usr/bin/env bash
# Prior and component ablations: networks pretrained from scratch for 30k steps with one change each,
# evaluated on the DEV and TEST data sets in all three regimes.
#
#   abl_reference  mixed prior (parametric + simulated committees), correction head, locality features
#   abl_paramonly  parametric committees only
#   abl_simonly    simulated committees only
#   abl_nores      without the class-equivariant correction head
#   abl_noloc      without the kNN locality features
#
# Usage: bash scripts/ablations.sh          (PY=/path/to/python to choose the interpreter)
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
MEM=${MEM_FRAC:-0.9}
mkdir -p runs/logs results

configs=(
  "abl_reference --residual --sim-frac 0.5"
  "abl_paramonly --residual --sim-frac 0.0"
  "abl_simonly   --residual --sim-frac 1.0"
  "abl_nores     --sim-frac 0.5"
  "abl_noloc     --residual --no-loc --sim-frac 0.5"
)
for cfg in "${configs[@]}"; do
  set -- $cfg; name=$1; shift
  echo "== $name"
  $PY scripts/train.py --name "$name" --steps 30000 --tokens 80000 --mem-frac "$MEM" --sim-pool runs/sim_pool "$@" \
      > "runs/logs/train_$name.log" 2>&1
  for split in dev test; do
    for c in committees_n128 committees_n512 committees; do
      $PY scripts/evaluate.py --split $split --committees $c --baselines --ckpt "runs/ckpt/$name.pt" \
          --out "results/${split}_fpfn_${name}_$c.csv" > "runs/logs/eval_${split}_${name}_$c.log" 2>&1
    done
  done
done

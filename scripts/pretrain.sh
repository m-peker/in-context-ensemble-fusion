#!/usr/bin/env bash
# Pretraining recipe of the released FusionPFN model (five stages, each warm-started from the previous one).
#
#   stage 1  parametric committee prior, no systematic confusions, pooling head only     20k steps
#   stage 2  + class-equivariant correction head, + systematic-confusion prior           15k steps
#   stage 3  + simulated committees (50% of the batches, needs runs/sim_pool)             15k steps
#   stage 4  continue, lr 2e-4                                                            30k steps
#   stage 5  continue, lr 1.5e-4                                                          60k steps
#
# Usage:  bash scripts/pretrain.sh [seed]          (default seed 42)
#         PY=/path/to/python bash scripts/pretrain.sh 43
# Every stage resumes from its checkpoint if interrupted. Result: runs/ckpt/fusionpfn_seed<seed>.pt
set -euo pipefail
S=${1:-42}
cd "$(dirname "$0")/.."
PY=${PY:-python}
MEM=${MEM_FRAC:-0.9}
mkdir -p runs/ckpt runs/logs
POOL="--sim-pool runs/sim_pool --sim-frac 0.5"
n() { echo "seed${S}_stage$1"; }

$PY scripts/train.py --name "$(n 1)" --seed "$S" --steps 20000 --p-confuse 0 --tokens 120000 --mem-frac "$MEM"
$PY scripts/train.py --name "$(n 2)" --seed "$S" --steps 15000 --residual --init "runs/ckpt/$(n 1).pt" \
    --tokens 80000 --mem-frac "$MEM"
$PY scripts/train.py --name "$(n 3)" --seed "$S" --steps 15000 --residual --init "runs/ckpt/$(n 2).pt" \
    --tokens 80000 --mem-frac "$MEM" $POOL
$PY scripts/train.py --name "$(n 4)" --seed "$S" --steps 30000 --residual --init "runs/ckpt/$(n 3).pt" \
    --tokens 80000 --mem-frac "$MEM" $POOL --lr 2e-4
$PY scripts/train.py --name "$(n 5)" --seed "$S" --steps 60000 --residual --init "runs/ckpt/$(n 4).pt" \
    --tokens 80000 --mem-frac "$MEM" $POOL --lr 1.5e-4

cp "runs/ckpt/$(n 5).pt" "runs/ckpt/fusionpfn_seed${S}.pt"
echo "done: runs/ckpt/fusionpfn_seed${S}.pt"

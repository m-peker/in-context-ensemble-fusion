#!/usr/bin/env bash
# Fetch the official implementation of Regularized Neural Ensemblers (Pineda Arango et al., AutoML 2025),
# used as a baseline. It is imported from external/ without installation, because its pinned dependencies
# would downgrade torch / scikit-learn.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p external
if [ ! -d external/RegularizedNeuralEnsemblers ]; then
  git clone https://github.com/machinelearningnuremberg/RegularizedNeuralEnsemblers.git external/RegularizedNeuralEnsemblers
fi
git -C external/RegularizedNeuralEnsemblers checkout 7fd4c410aaa4c2a7cf81d14172c8459609b1ce9d
echo "Neural Ensemblers ready in external/RegularizedNeuralEnsemblers"

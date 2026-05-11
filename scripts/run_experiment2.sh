#!/usr/bin/env bash
set -euo pipefail
python experiments/run_experiment2_graph_posterior.py \
  --m 15 \
  --d 3 \
  --n_chains 4 \
  --n_steps 6000 \
  --burn 1000 \
  --outdir outputs/experiment2_graph_posterior

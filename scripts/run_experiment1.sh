#!/usr/bin/env bash
set -euo pipefail
python experiments/run_experiment1_spd_posterior.py \
  --d 3 \
  --n_chains 4 \
  --N 15000 \
  --burn 3000 \
  --thin 5 \
  --outdir outputs/experiment1_spd_posterior

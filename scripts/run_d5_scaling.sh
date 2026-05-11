#!/usr/bin/env bash
set -euo pipefail

python experiments/run_d5_m_sweep_graph_posterior.py \
  --m_list 20 \
  --n_chains 4 \
  --n_steps 16000 \
  --burn 4000 \
  --h_cone 8e-3 \
  --h_euclid 4e-5 \
  --h_rmala 3e-4 \
  --outdir outputs/cone_mala_d5_m_sweep

python experiments/run_d5_m_sweep_graph_posterior.py \
  --m_list 50 \
  --n_chains 4 \
  --n_steps 16000 \
  --burn 4000 \
  --h_cone 6e-3 \
  --h_euclid 2e-5 \
  --h_rmala 1e-4 \
  --outdir outputs/cone_mala_d5_m_sweep

python experiments/run_d5_m_sweep_graph_posterior.py \
  --m_list 100 \
  --n_chains 4 \
  --n_steps 8000 \
  --burn 2000 \
  --h_cone 4e-3 \
  --h_euclid 2e-5 \
  --h_rmala 5e-5 \
  --outdir outputs/cone_mala_d5_m_sweep

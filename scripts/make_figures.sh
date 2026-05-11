#!/usr/bin/env bash
set -euo pipefail
python experiments/sensitivity_validation_fig1.py
python plotting/plot_scaling_d5.py --outdir results --plotdir figures --m_list 20,50,100 --use_phi_only

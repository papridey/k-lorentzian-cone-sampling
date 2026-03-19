# k-lorentzian-cone-sampling

Reference code for the SPD(3) worked example in the manuscript  
**``Cone-Induced Geometry for Sampling on Cones via K-Lorentzian Polynomials''**.

This repository implements two Metropolis–Hastings kernels that target the same **intrinsic Riemannian law**
on the SPD cone with the affine-invariant metric:

π(dX) ∝ exp(−Φ(X)) · vol_g(dX),

where vol_g is the Riemannian volume measure.  
Importantly, proposal densities are evaluated **with respect to vol_g (not Lebesgue)** via the exponential-map
Jacobian correction.

## What you get
Running the script reproduces the main diagnostics and plots used in the paper:
- Acceptance rate
- split-Rhat (convergence across chains)
- ESS/sec (efficiency)
- pooled marginal summaries (means/SD/quantiles)
- MCSE + z-score mean agreement across methods
- an empirical Poincaré / rho proxy (rho_hat)

## Repository layout
- `src/spd_ai_geometry.py` — SPD helpers; affine-invariant log/exp maps; log Jacobian term log J.
- `src/targets.py` — target energy Φ and gradients for the intrinsic target.
- `src/samplers.py` — `geom_MALA` (geometry-aware) and `naive_Euclid_drift_in_S` baseline.
- `src/diagnostics.py` — split-Rhat, ESS, MCSE, z-scores, rho_hat proxy.
- `src/plotting.py` — trace plots and cross-method histogram/ECDF overlays.
- `src/scripts/run_spd3_riem_target_mcse.py` — main script reproducing the SPD(3) experiment.

## Output
Running the script creates a timestamped output directory containing:
- `results/summary.csv` (acceptance, split-Rhat, ESS/sec)
- `results/pooled_method_summary_stats.csv` (pooled marginal summaries)
- `results/rho_proxy.csv` (empirical Poincaré / rho proxy)
- `results/mcse_zscore_comparison.csv` (MCSE + z-score mean agreement)
- `plots/` (trace plots + cross-method histogram/ECDF overlays)

## Citation
If you use this code, please cite the repository (see `CITATION.cff`).

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.scripts.run_spd3_riem_target_mcse

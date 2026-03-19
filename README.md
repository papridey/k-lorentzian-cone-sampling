# k-lorentzian-cone-sampling

Code accompanying the manuscript ``Cone-Induced Geometry for Sampling on Cones via $\K$-Lorentzian Polynomials''.  
This repository implements geometry-aware Metropolis–Hastings samplers on the SPD cone using the affine-invariant metric, and reproduces the diagnostic tables/plots reported in the paper (acceptance, split-\hat{R}, ESS/sec, pooled marginals, MCSE/z-scores, and an empirical Poincaré/\rho proxy).

---

## Contents

- `src/spd_ai_geometry.py` — SPD helpers, affine-invariant log/exp maps, and exponential-map Jacobian term.
- `src/targets.py` — target energy \(\Phi\) and associated gradients for the intrinsic target \(\pi \propto e^{-\Phi}\,\mathrm{vol}_g\).
- `src/samplers.py` — `geom_MALA` (geometry-aware) and `naive_Euclid_drift_in_S` baseline.
- `src/diagnostics.py` — split-\(\hat{R}\), ESS, MCSE, z-scores, and \(\widehat{\rho}\) proxy.
- `src/plotting.py` — trace plots and cross-method histogram/ECDF overlays.
- `src/scripts/run_spd3_riem_target_mcse.py` — main experiment script used for the SPD(3) worked example.

---

## Installation

### Requirements
- Python >= 3.10
- NumPy, SciPy, Pandas, Matplotlib

Install dependencies:
```bash
pip install -r requirements.txt

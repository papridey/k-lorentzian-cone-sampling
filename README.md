# Cone-induced geometry-aware sampling on SPD(3)

Code accompanying the manuscript ``Cone-Induced Geometry for Sampling on Cones via $\K$-Lorentzian Polynomials''.  

This repository contains the reference implementation for the **SPD(3) worked example**:
two Metropolis–Hastings kernels targeting the same **intrinsic Riemannian law**
\[
\pi(dX)\propto \exp(-\Phi(X))\,\mathrm{vol}_g(dX),
\]
on the SPD cone \(\mathbb S_{++}^3\) with the **affine-invariant metric**, including the **exponential-map Jacobian correction** in the proposal density (i.e., proposals are evaluated w.r.t. \(\mathrm{vol}_g\), not Lebesgue).

The code produces the diagnostics and plots used in the paper: acceptance, split-\(\widehat R\), ESS/sec, pooled marginal summaries, MCSE/z-score mean agreement, and an empirical Poincaré/\(\widehat\rho\) proxy.

---

## Repository layout

- `src/spd_ai_geometry.py` — SPD helpers, affine-invariant log/exp maps, and \(\log J\) (Exp Jacobian term).
- `src/targets.py` — the target energy \(\Phi\) and gradients.
- `src/samplers.py` — `geom_MALA` and `naive_Euclid_drift_in_S`.
- `src/diagnostics.py` — split-\(\widehat R\), ESS, MCSE, z-scores, \(\widehat\rho\) proxy.
- `src/plotting.py` — trace plots + cross-method histogram/ECDF overlays.
- `src/scripts/run_spd3_riem_target_mcse.py` — main script reproducing the SPD(3) experiment.

---

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
python -m src.scripts.run_spd3_riem_target_mcse

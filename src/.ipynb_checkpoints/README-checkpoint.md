# Cone-induced geometry-aware sampling on SPD(3)

This repository contains the reference implementation for the SPD(3) worked example:
two Metropolis–Hastings kernels targeting the same Riemannian density
\[
\pi(dX)\propto \exp(-\Phi(X))\,\mathrm{vol}_g(dX),
\]
on the SPD cone with the affine-invariant metric, including the exponential-map Jacobian correction.

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
python scripts/run_spd3_riem_target_mcse.py
"""
Cone-induced geometry-aware sampling utilities.

This package provides:
- Affine-invariant SPD geometry + exponential-map Jacobian (spd_ai_geometry.py)
- Target definition for SPD(3) Riemannian density (targets.py)
- MH samplers targeting pi(dX) ∝ exp(-Phi(X)) vol_g(dX) (samplers.py)
- Diagnostics + rho proxy + MCSE/z-score checks (diagnostics.py)
- Plotting helpers (plotting.py)
"""
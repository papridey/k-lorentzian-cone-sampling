import numpy as np

from .spd_ai_geometry import (
    sym,
    spd_project,
    spd_sqrt_and_invsqrt,
    logdet_spd,
    affine_invariant_dist2,
    ai_log_map_U,
)

class TargetSPD_AI:
    r"""
    Riemannian target on SPD(3) with affine-invariant base measure:

      π(dX) ∝ exp(-Φ(X)) vol_g(dX)

    where
      Φ(X) = (λ/2) d_g(X, X0)^2 - β logdet(X) + (κ/2)(tr(X)-1)^2.

    Manuscript-style ell = -Φ + 1/2 logdet G:
    in whitened S-coordinates for the affine-invariant metric, det(G) is constant,
    so logdetG_half is set to 0.0 (and would cancel in MH ratios anyway if consistent).

    Riemannian gradient (affine-invariant metric):
      grad_g (λ/2 d^2) = -λ Log_X(X0)
      grad_g (-β logdet) = -β X
      penalty f=(κ/2)(tr-1)^2:
        grad_E f = κ(tr-1) I
        grad_g f = X (grad_E f) X = κ(tr-1) X^2
    """
    def __init__(self, lambda_: float = 10.0, beta: float = 1.0, kappa: float = 50.0, X0=None):
        self.lambda_ = float(lambda_)
        self.beta = float(beta)
        self.kappa = float(kappa)
        self.X0 = spd_project(np.eye(3) if X0 is None else X0)

    def Phi(self, X: np.ndarray) -> float:
        X = spd_project(X)
        d2 = affine_invariant_dist2(X, self.X0)
        ld = logdet_spd(X)
        tr = float(np.trace(X))
        return 0.5 * self.lambda_ * d2 - self.beta * ld + 0.5 * self.kappa * (tr - 1.0) ** 2

    def logdetG_half(self, X: np.ndarray) -> float:
        # Constant in whitened tangent coordinates for affine-invariant SPD metric.
        return 0.0

    def ell(self, X: np.ndarray) -> float:
        # Lebesgue log-density used in algorithm pseudocode; constant omitted.
        return -self.Phi(X) + self.logdetG_half(X)

    def grad_g_Phi_U(self, X: np.ndarray) -> np.ndarray:
        X = spd_project(X)
        Udist = ai_log_map_U(X, self.X0)      # Log_X(X0)
        Gg = -self.lambda_ * Udist
        Gg += -self.beta * X                  # grad_g(-β logdet) = -β X
        tr = float(np.trace(X))
        Gg += self.kappa * (tr - 1.0) * sym(X @ X)  # penalty grad_g
        return sym(Gg)

    def grad_E_Phi(self, X: np.ndarray) -> np.ndarray:
        # grad_E = X^{-1} grad_g X^{-1}
        X = spd_project(X)
        Gg = self.grad_g_Phi_U(X)
        _, Xis = spd_sqrt_and_invsqrt(X)
        return sym(Xis @ Gg @ Xis)
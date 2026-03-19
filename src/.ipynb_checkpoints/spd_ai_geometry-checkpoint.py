import math
import numpy as np
from numpy.linalg import eigvalsh
from scipy.linalg import eigh, expm


# ============================================================
# Basic SPD helpers
# ============================================================

def sym(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)

def fro_norm2(A: np.ndarray) -> float:
    return float(np.sum(A * A))

def spd_project(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Symmetrize + eigen-floor to SPD."""
    X = sym(X)
    w, V = eigh(X)
    w = np.maximum(w, eps)
    return sym((V * w) @ V.T)

def spd_sqrt_and_invsqrt(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return X^{1/2}, X^{-1/2} for SPD X."""
    X = sym(X)
    w, V = eigh(X)
    w = np.maximum(w, 1e-15)
    sw = np.sqrt(w)
    isw = 1.0 / sw
    Xs = sym((V * sw) @ V.T)
    Xis = sym((V * isw) @ V.T)
    return Xs, Xis

def logdet_spd(X: np.ndarray) -> float:
    w = np.maximum(eigvalsh(sym(X)), 1e-15)
    return float(np.sum(np.log(w)))


# ============================================================
# Affine-invariant SPD geometry (whitened S-coordinates)
# ============================================================

def ai_log_Scoords(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    S = log( X^{-1/2} Y X^{-1/2} ) (symmetric).
    """
    _, Xis = spd_sqrt_and_invsqrt(X)
    M = sym(Xis @ Y @ Xis)
    w, V = eigh(M)
    w = np.maximum(w, 1e-15)
    return sym((V * np.log(w)) @ V.T)

def ai_exp_from_S(X: np.ndarray, S: np.ndarray) -> np.ndarray:
    """
    Y = X^{1/2} exp(S) X^{1/2}, then project to SPD.
    """
    Xs, _ = spd_sqrt_and_invsqrt(X)
    Y = sym(Xs @ expm(sym(S)) @ Xs)
    return spd_project(Y)

def ai_log_map_U(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    U = Log_X(Y) in ambient symmetric coordinates:
      U = X^{1/2} S X^{1/2}.
    """
    S = ai_log_Scoords(X, Y)
    Xs, _ = spd_sqrt_and_invsqrt(X)
    return sym(Xs @ S @ Xs)

def affine_invariant_dist2(X: np.ndarray, Y: np.ndarray) -> float:
    """d_g(X,Y)^2 = || log( X^{-1/2} Y X^{-1/2} ) ||_F^2."""
    S = ai_log_Scoords(X, Y)
    return fro_norm2(S)


# ============================================================
# Exponential-map Jacobian on SPD with affine-invariant metric
# ============================================================

def log_J_exp_spd(S: np.ndarray, tol: float = 1e-10) -> float:
    """
    log J(S) where
      J(S)=∏_{i<j} sinh((s_i-s_j)/2)/((s_i-s_j)/2),
    with (s_i) eigenvalues of S. Uses continuous extension near 0.
    """
    s = np.linalg.eigvalsh(sym(S))
    logJ = 0.0
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            a = 0.5 * (s[i] - s[j])
            if abs(a) < tol:
                logJ += (a * a) / 6.0  # log(sinh(a)/a) ≈ a^2/6
            else:
                logJ += math.log(abs(math.sinh(a) / a))
    return float(logJ)


# ============================================================
# Gaussian utilities in symmetric space
# ============================================================

def sample_sym_gaussian(rng: np.random.Generator, d: int = 3) -> np.ndarray:
    A = rng.normal(size=(d, d))
    return sym(A)

def log_gauss_sym(S: np.ndarray, mean: np.ndarray, sigma2: float) -> float:
    """
    Frobenius Gaussian log-density up to an additive constant:
      log ϕ(S; mean, sigma2 I) = -||S-mean||_F^2/(2 sigma2) + const.
    """
    D = sym(S - mean)
    return -0.5 * fro_norm2(D) / sigma2
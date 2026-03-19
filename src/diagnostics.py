import numpy as np
import pandas as pd

from .spd_ai_geometry import (
    sym,
    fro_norm2,
    spd_sqrt_and_invsqrt,
    logdet_spd,
    affine_invariant_dist2,
    ai_log_map_U,
)

# ============================================================
# Split-Rhat, ESS
# ============================================================

def split_rhat(chains_1d: list[np.ndarray]) -> float:
    """Split-Rhat for a list of 1D numpy arrays (one per chain)."""
    split = []
    for c in chains_1d:
        T = len(c)
        h = T // 2
        split.append(c[:h])
        split.append(c[h : 2 * h])
    split = np.array(split)
    _, n = split.shape

    means = split.mean(axis=1)
    vars_ = split.var(axis=1, ddof=1)
    W = vars_.mean()
    B = n * means.var(ddof=1)
    var_hat = (n - 1) / n * W + (1 / n) * B
    return float(np.sqrt(var_hat / W))

def ess_1d(x: np.ndarray) -> float:
    """
    Simple initial-positive-sequence ESS estimator via autocorrelation.
    Conservative and stable for reporting.
    """
    x = np.asarray(x)
    x = x - x.mean()
    n = len(x)
    if n < 10:
        return float(n)
    var = np.dot(x, x) / n
    if var <= 1e-30:
        return float(n)

    max_lag = min(2000, n - 1)
    acf = np.empty(max_lag + 1)
    acf[0] = 1.0
    for lag in range(1, max_lag + 1):
        acf[lag] = np.dot(x[: n - lag], x[lag:]) / (n - lag) / var

    tau = 1.0
    for k in range(1, max_lag, 2):
        s = acf[k] + acf[k + 1]
        if s <= 0:
            break
        tau += 2 * s
    return float(max(1.0, min(n, n / tau)))


# ============================================================
# rho proxy (empirical Poincare proxy) under affine-invariant metric
# ============================================================

def ai_norm2(X: np.ndarray, U: np.ndarray) -> float:
    """||U||_g^2 = tr(X^{-1} U X^{-1} U) = || X^{-1/2} U X^{-1/2} ||_F^2."""
    _, Xis = spd_sqrt_and_invsqrt(X)
    A = sym(Xis @ U @ Xis)
    return fro_norm2(A)

def grad_g_logdet(X: np.ndarray) -> np.ndarray:
    # grad_g logdet = X
    return sym(X)

def grad_g_half_d2(X: np.ndarray, X0: np.ndarray) -> np.ndarray:
    # grad_g (1/2 d^2) = -Log_X(X0)
    return sym(-ai_log_map_U(X, X0))

def grad_g_trCX(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    # grad_g tr(CX) = X C X
    return sym(X @ C @ X)

def rho_hat(samples: np.ndarray, vals: np.ndarray, grad_list: list[np.ndarray]) -> float:
    """
    \hat rho(h) = (mean ||grad_g h||_g^2) / Var(h).
    """
    vals = np.asarray(vals)
    v = float(np.var(vals, ddof=1))
    if v <= 1e-30:
        return np.nan
    g2 = np.array([ai_norm2(X, U) for X, U in zip(samples, grad_list)], dtype=float)
    return float(np.mean(g2) / v)

def compute_rho_suite(pooled_samples: np.ndarray, X0: np.ndarray, n_linear_tests: int = 5, seed: int = 123) -> dict:
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}

    # logdet
    vals = np.array([logdet_spd(X) for X in pooled_samples])
    grads = [grad_g_logdet(X) for X in pooled_samples]
    out["rho_logdet"] = rho_hat(pooled_samples, vals, grads)

    # 0.5 d^2
    vals = np.array([0.5 * affine_invariant_dist2(X, X0) for X in pooled_samples])
    grads = [grad_g_half_d2(X, X0) for X in pooled_samples]
    out["rho_half_d2"] = rho_hat(pooled_samples, vals, grads)

    # linear tests tr(CX)
    for t in range(n_linear_tests):
        C = sym(rng.normal(size=(3, 3)))
        vals = np.array([float(np.trace(C @ X)) for X in pooled_samples])
        grads = [grad_g_trCX(X, C) for X in pooled_samples]
        out[f"rho_trCX_{t}"] = rho_hat(pooled_samples, vals, grads)

    finite = [v for v in out.values() if np.isfinite(v)]
    out["rho_min"] = float(np.min(finite)) if finite else np.nan
    return out


# ============================================================
# MCSE + z-score comparison for pooled means
# ============================================================

def pooled_series(chains_stats_list: list[pd.DataFrame], col: str) -> np.ndarray:
    return np.concatenate([df[col].values for df in chains_stats_list], axis=0)

def pooled_ess_from_chains(chains_stats_list: list[pd.DataFrame], col: str) -> float:
    # conservative: sum per-chain ESS
    return float(sum(ess_1d(df[col].values) for df in chains_stats_list))

def mcse(sd: float, ess: float) -> float:
    return float(sd / np.sqrt(max(1.0, ess)))

def build_mcse_z_table(chains_stats_dict: dict, methodA: str, methodB: str, cols: list[str]) -> pd.DataFrame:
    A = chains_stats_dict[methodA]
    B = chains_stats_dict[methodB]
    rows = []
    for col in cols:
        xA = pooled_series(A, col)
        xB = pooled_series(B, col)

        meanA = float(np.mean(xA))
        meanB = float(np.mean(xB))
        diff = float(meanA - meanB)

        essA = pooled_ess_from_chains(A, col)
        essB = pooled_ess_from_chains(B, col)

        sdA = float(np.std(xA, ddof=1))
        sdB = float(np.std(xB, ddof=1))

        mcseA = mcse(sdA, essA)
        mcseB = mcse(sdB, essB)

        z = diff / np.sqrt(mcseA**2 + mcseB**2 + 1e-30)

        rows.append({
            "observable": col,
            f"mean_{methodA}": meanA,
            f"mean_{methodB}": meanB,
            "diff": diff,
            f"ESS_{methodA}": essA,
            f"ESS_{methodB}": essB,
            f"MCSE_{methodA}": mcseA,
            f"MCSE_{methodB}": mcseB,
            "z_score": float(z),
        })
    return pd.DataFrame(rows)
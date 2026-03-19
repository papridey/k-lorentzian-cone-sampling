import math
import numpy as np
import pandas as pd
from numpy.linalg import eigvalsh

from .spd_ai_geometry import (
    sym,
    spd_project,
    spd_sqrt_and_invsqrt,
    logdet_spd,
    ai_log_Scoords,
    ai_exp_from_S,
    affine_invariant_dist2,
    log_J_exp_spd,
    log_gauss_sym,
    sample_sym_gaussian,
)

def run_chain_geom_MALA(
    rng: np.random.Generator,
    target,
    N: int = 15000,
    burn: int = 3000,
    thin: int = 5,
    h: float = 8e-3,
    X_init=None,
    store_mats: bool = True,
):
    """
    Geometry-aware MH kernel targeting π(dX) ∝ exp(-Φ(X)) vol_g(dX).

    Proposal in whitened tangent coords S:
      mean_S = -h * (X^{-1/2} grad_g Φ(X) X^{-1/2})
      S ~ N(mean_S, 2h I)
      Y = Exp_X(S) = X^{1/2} exp(S) X^{1/2}

    Proposal density w.r.t vol_g:
      log q_vol = log N(S; mean_S, 2h I) - log J(S),
    where J is the Exp Jacobian in S-coordinates.
    """
    if X_init is None:
        X = spd_project(np.eye(3) + 0.20 * rng.normal(size=(3, 3)))
    else:
        X = spd_project(X_init)

    sigma2 = 2.0 * h
    accepts = 0
    stats = []
    kept_X = []
    t0 = __import__("time").time()

    for k in range(N):
        ell_x = target.ell(X)

        Gg = target.grad_g_Phi_U(X)
        _, Xis = spd_sqrt_and_invsqrt(X)
        gradS = sym(Xis @ Gg @ Xis)
        mean_S = sym(-h * gradS)

        Z = sample_sym_gaussian(rng, 3)
        S = sym(mean_S + math.sqrt(2.0 * h) * Z)
        Y = ai_exp_from_S(X, S)
        ell_y = target.ell(Y)

        S_bwd = ai_log_Scoords(Y, X)

        Gg_y = target.grad_g_Phi_U(Y)
        _, Yis = spd_sqrt_and_invsqrt(Y)
        gradS_y = sym(Yis @ Gg_y @ Yis)
        mean_S_y = sym(-h * gradS_y)

        logq_xy = log_gauss_sym(S, mean_S, sigma2) - log_J_exp_spd(S)
        logq_yx = log_gauss_sym(S_bwd, mean_S_y, sigma2) - log_J_exp_spd(S_bwd)

        logr = (ell_y + logq_yx) - (ell_x + logq_xy)
        if np.log(rng.uniform()) < min(0.0, logr):
            X = Y
            accepts += 1

        if k >= burn and ((k - burn) % thin == 0):
            ev = eigvalsh(X)
            stats.append(
                {
                    "logdet": logdet_spd(X),
                    "lmin": float(ev.min()),
                    "lmax": float(ev.max()),
                    "d2": affine_invariant_dist2(X, target.X0),
                    "tr": float(np.trace(X)),
                }
            )
            if store_mats:
                kept_X.append(X.copy())

    meta = {"acc_rate": accepts / float(N), "elapsed": __import__("time").time() - t0}
    return np.array(kept_X), pd.DataFrame(stats), meta


def run_chain_naive_Euclid_drift_in_S(
    rng: np.random.Generator,
    target,
    N: int = 15000,
    burn: int = 3000,
    thin: int = 5,
    h: float = 8e-3,
    X_init=None,
    store_mats: bool = True,
):
    """
    Baseline MH kernel with the SAME target π(dX) ∝ exp(-Φ(X)) vol_g(dX),
    but using a naive drift constructed from the Euclidean gradient mapped to S-coordinates:
      mean_S = -h * (X^{-1/2} grad_E Φ(X) X^{-1/2}).

    Proposal and MH ratio still use log q_vol = log N - log J, i.e., correct base measure vol_g.
    """
    if X_init is None:
        X = spd_project(np.eye(3) + 0.20 * rng.normal(size=(3, 3)))
    else:
        X = spd_project(X_init)

    sigma2 = 2.0 * h
    accepts = 0
    stats = []
    kept_X = []
    t0 = __import__("time").time()

    for k in range(N):
        ell_x = target.ell(X)

        GE = target.grad_E_Phi(X)
        _, Xis = spd_sqrt_and_invsqrt(X)
        gradS = sym(Xis @ GE @ Xis)
        mean_S = sym(-h * gradS)

        Z = sample_sym_gaussian(rng, 3)
        S = sym(mean_S + math.sqrt(2.0 * h) * Z)
        Y = ai_exp_from_S(X, S)
        ell_y = target.ell(Y)

        S_bwd = ai_log_Scoords(Y, X)

        GE_y = target.grad_E_Phi(Y)
        _, Yis = spd_sqrt_and_invsqrt(Y)
        gradS_y = sym(Yis @ GE_y @ Yis)
        mean_S_y = sym(-h * gradS_y)

        logq_xy = log_gauss_sym(S, mean_S, sigma2) - log_J_exp_spd(S)
        logq_yx = log_gauss_sym(S_bwd, mean_S_y, sigma2) - log_J_exp_spd(S_bwd)

        logr = (ell_y + logq_yx) - (ell_x + logq_xy)
        if np.log(rng.uniform()) < min(0.0, logr):
            X = Y
            accepts += 1

        if k >= burn and ((k - burn) % thin == 0):
            ev = eigvalsh(X)
            stats.append(
                {
                    "logdet": logdet_spd(X),
                    "lmin": float(ev.min()),
                    "lmax": float(ev.max()),
                    "d2": affine_invariant_dist2(X, target.X0),
                    "tr": float(np.trace(X)),
                }
            )
            if store_mats:
                kept_X.append(X.copy())

    meta = {"acc_rate": accepts / float(N), "elapsed": __import__("time").time() - t0}
    return np.array(kept_X), pd.DataFrame(stats), meta

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "plotting"))

import numpy as np
import pandas as pd
from numpy.linalg import eigvalsh
from scipy.linalg import eigh, expm

from plotting import (
    compute_pooled_summary,
    save_chain_trace,
    save_method_ecdf_overlay,
    save_method_hist_overlay,
    save_two_panel_ecdf,
)


def sym(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)


def fro_norm2(A: np.ndarray) -> float:
    return float(np.sum(A * A))


def spd_project(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = sym(X)
    w, V = eigh(X)
    w = np.maximum(w, eps)
    return sym((V * w) @ V.T)


def is_spd(X: np.ndarray, eps: float = 1e-12) -> bool:
    return bool(np.min(eigvalsh(sym(X))) > eps)


def spd_sqrt_and_invsqrt(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = sym(X)
    w, V = eigh(X)
    w = np.maximum(w, 1e-15)
    sw = np.sqrt(w)
    isw = 1.0 / sw
    Xs = sym((V * sw) @ V.T)
    Xis = sym((V * isw) @ V.T)
    return Xs, Xis


def spd_inv(X: np.ndarray) -> np.ndarray:
    _, Xis = spd_sqrt_and_invsqrt(X)
    return sym(Xis @ Xis)


def logdet_spd(X: np.ndarray) -> float:
    w = np.maximum(eigvalsh(sym(X)), 1e-15)
    return float(np.sum(np.log(w)))


def ai_log_congruence_coords(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    _, Xis = spd_sqrt_and_invsqrt(X)
    M = sym(Xis @ Y @ Xis)
    w, V = eigh(M)
    w = np.maximum(w, 1e-15)
    return sym((V * np.log(w)) @ V.T)


def ai_exp_from_congruence(X: np.ndarray, S: np.ndarray) -> np.ndarray:
    Xs, _ = spd_sqrt_and_invsqrt(X)
    return spd_project(sym(Xs @ expm(sym(S)) @ Xs))


def ai_log_map_U(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    S = ai_log_congruence_coords(X, Y)
    Xs, _ = spd_sqrt_and_invsqrt(X)
    return sym(Xs @ S @ Xs)


def affine_invariant_dist2(X: np.ndarray, Y: np.ndarray) -> float:
    return fro_norm2(ai_log_congruence_coords(X, Y))


def log_J_exp_spd(S: np.ndarray, tol: float = 1e-10) -> float:
    s = np.linalg.eigvalsh(sym(S))
    logJ = 0.0
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            a = 0.5 * (s[i] - s[j])
            if abs(a) < tol:
                logJ += (a * a) / 6.0
            else:
                logJ += math.log(abs(math.sinh(a) / a))
    return float(logJ)


def sym_dim(d: int) -> int:
    return d * (d + 1) // 2


def sym_to_orthovec(A: np.ndarray) -> np.ndarray:
    A = sym(A)
    d = A.shape[0]
    out = []
    for i in range(d):
        out.append(A[i, i])
    for i in range(d):
        for j in range(i + 1, d):
            out.append(math.sqrt(2.0) * A[i, j])
    return np.array(out, dtype=float)


def orthovec_to_sym(v: np.ndarray, d: int) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(-1)
    A = np.zeros((d, d), dtype=float)
    idx = 0
    for i in range(d):
        A[i, i] = v[idx]
        idx += 1
    for i in range(d):
        for j in range(i + 1, d):
            A[i, j] = v[idx] / math.sqrt(2.0)
            A[j, i] = A[i, j]
            idx += 1
    return sym(A)


def random_sym_normal(d: int, rng: np.random.Generator) -> np.ndarray:
    return orthovec_to_sym(rng.normal(size=sym_dim(d)), d)


def orthonormal_basis_sym(d: int) -> list[np.ndarray]:
    basis = []
    for i in range(d):
        A = np.zeros((d, d), dtype=float)
        A[i, i] = 1.0
        basis.append(A)
    for i in range(d):
        for j in range(i + 1, d):
            A = np.zeros((d, d), dtype=float)
            A[i, j] = 1.0 / math.sqrt(2.0)
            A[j, i] = 1.0 / math.sqrt(2.0)
            basis.append(A)
    return basis


def metric_matrix_sym_affine_invariant(X: np.ndarray) -> np.ndarray:
    basis = orthonormal_basis_sym(X.shape[0])
    Xinv = spd_inv(X)
    p = len(basis)
    G = np.empty((p, p), dtype=float)
    for i, Ei in enumerate(basis):
        for j, Ej in enumerate(basis):
            G[i, j] = float(np.trace(Xinv @ Ei @ Xinv @ Ej))
    return sym(G)


def log_gauss_sym(S: np.ndarray, mean: np.ndarray, sigma2: float) -> float:
    D = sym(S - mean)
    p = sym_dim(S.shape[0])
    return float(-0.5 * p * math.log(2.0 * math.pi * sigma2) - 0.5 * fro_norm2(D) / sigma2)


def log_gauss_vec_iso(y: np.ndarray, mean: np.ndarray, sigma2: float) -> float:
    p = len(mean)
    diff = np.asarray(y) - np.asarray(mean)
    return float(-0.5 * (p * math.log(2.0 * math.pi * sigma2) + diff.dot(diff) / sigma2))


def log_gauss_vec_metric(y: np.ndarray, mean: np.ndarray, G: np.ndarray, sigma2_scale: float) -> float:
    diff = np.asarray(y) - np.asarray(mean)
    sign, logdetG = np.linalg.slogdet(G)
    if sign <= 0:
        return -np.inf
    p = len(mean)
    quad = float(diff @ G @ diff) / sigma2_scale
    return float(-0.5 * (p * math.log(2.0 * math.pi * sigma2_scale) - logdetG + quad))


def random_spd_around(rng: np.random.Generator, X0: np.ndarray, sigma: float = 0.35) -> np.ndarray:
    Xs, _ = spd_sqrt_and_invsqrt(X0)
    S = sigma * random_sym_normal(X0.shape[0], rng)
    Y = sym(Xs @ expm(sym(S)) @ Xs)
    return spd_project(Y)


def karcher_mean_ai(data: list[np.ndarray], max_iter: int = 50, tol: float = 1e-10) -> np.ndarray:
    X = spd_project(np.mean(data, axis=0))
    for _ in range(max_iter):
        Xs, _ = spd_sqrt_and_invsqrt(X)
        G = np.zeros_like(X)
        for S in data:
            G += ai_log_congruence_coords(X, S)
        G /= len(data)
        if fro_norm2(G) < tol:
            break
        X = spd_project(sym(Xs @ expm(sym(G)) @ Xs))
    return X


class TargetSPD_CovData:
    def __init__(self, data_list: list[np.ndarray], lambda_: float = 4.0, beta: float = 1.0, kappa: float = 20.0, c_trace: float | None = None):
        self.data = [spd_project(S) for S in data_list]
        self.lambda_ = float(lambda_)
        self.beta = float(beta)
        self.kappa = float(kappa)
        self.Xbar = karcher_mean_ai(self.data)
        self.c_trace = float(np.trace(self.Xbar) if c_trace is None else c_trace)
        self.d = self.data[0].shape[0]
        self.riem_logvol_coeff = -0.5 * (self.d + 1.0)

    def Phi(self, X: np.ndarray) -> float:
        X = spd_project(X)
        fit = 0.5 * self.lambda_ * sum(affine_invariant_dist2(X, S) for S in self.data)
        ld = logdet_spd(X)
        tr = float(np.trace(X))
        return fit - self.beta * ld + 0.5 * self.kappa * (tr - self.c_trace) ** 2

    def ell_intrinsic_chart(self, X: np.ndarray) -> float:
        return -self.Phi(X)

    def logpi_leb(self, X: np.ndarray) -> float:
        return -self.Phi(X) + self.riem_logvol_coeff * logdet_spd(X)

    def grad_g_Phi_U(self, X: np.ndarray) -> np.ndarray:
        X = spd_project(X)
        Gg = np.zeros_like(X)
        for S in self.data:
            Gg += -self.lambda_ * ai_log_map_U(X, S)
        Gg += -self.beta * X
        tr = float(np.trace(X))
        Gg += self.kappa * (tr - self.c_trace) * sym(X @ X)
        return sym(Gg)

    def grad_E_Phi(self, X: np.ndarray) -> np.ndarray:
        Gg = self.grad_g_Phi_U(X)
        Xinv = spd_inv(X)
        return sym(Xinv @ Gg @ Xinv)

    def grad_E_logpi_leb(self, X: np.ndarray) -> np.ndarray:
        Xinv = spd_inv(X)
        return sym(-self.grad_E_Phi(X) + self.riem_logvol_coeff * Xinv)


def record_stats(X: np.ndarray, target: TargetSPD_CovData) -> dict[str, float]:
    ev = eigvalsh(X)
    return {
        "logdet": logdet_spd(X),
        "lmin": float(ev.min()),
        "lmax": float(ev.max()),
        "d2_bar": affine_invariant_dist2(X, target.Xbar),
        "tr": float(np.trace(X)),
        "Phi": float(target.Phi(X)),
    }


def run_chain_cone_geom_mala(rng, target, N=15000, burn=3000, thin=5, h=8e-3, X_init=None, store_mats=True):
    d = target.d
    X = spd_project(np.eye(d) + 0.20 * rng.normal(size=(d, d))) if X_init is None else spd_project(X_init)
    sigma2 = 2.0 * h
    accepts, stats, kept_X = 0, [], []
    t0 = time.time()

    for k in range(N):
        ell_x = target.ell_intrinsic_chart(X)
        Gg = target.grad_g_Phi_U(X)
        _, Xis = spd_sqrt_and_invsqrt(X)
        gradS = sym(Xis @ Gg @ Xis)
        mean_S = sym(-h * gradS)

        Z = random_sym_normal(d, rng)
        S = sym(mean_S + math.sqrt(2.0 * h) * Z)
        Y = ai_exp_from_congruence(X, S)
        ell_y = target.ell_intrinsic_chart(Y)

        S_bwd = ai_log_congruence_coords(Y, X)
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
            stats.append(record_stats(X, target))
            if store_mats:
                kept_X.append(X.copy())

    return np.array(kept_X), pd.DataFrame(stats), {"acc_rate": accepts / float(N), "elapsed": time.time() - t0}


def run_chain_euclidean_mala(rng, target, N=15000, burn=3000, thin=5, h=8e-4, X_init=None, store_mats=True):
    d = target.d
    X = spd_project(np.eye(d) + 0.20 * rng.normal(size=(d, d))) if X_init is None else spd_project(X_init)
    sigma2 = 2.0 * h
    accepts, stats, kept_X = 0, [], []
    t0 = time.time()

    for k in range(N):
        logpi_x = target.logpi_leb(X)
        x = sym_to_orthovec(X)
        gradx = sym_to_orthovec(target.grad_E_logpi_leb(X))
        mean_x = x + h * gradx
        y = mean_x + math.sqrt(2.0 * h) * rng.normal(size=len(x))
        Y = orthovec_to_sym(y, d)

        if is_spd(Y):
            logpi_y = target.logpi_leb(Y)
            grad_y = sym_to_orthovec(target.grad_E_logpi_leb(Y))
            mean_y = y + h * grad_y
            logq_xy = log_gauss_vec_iso(y, mean_x, sigma2)
            logq_yx = log_gauss_vec_iso(x, mean_y, sigma2)
            logr = (logpi_y + logq_yx) - (logpi_x + logq_xy)
            if np.log(rng.uniform()) < min(0.0, logr):
                X = spd_project(Y)
                accepts += 1

        if k >= burn and ((k - burn) % thin == 0):
            stats.append(record_stats(X, target))
            if store_mats:
                kept_X.append(X.copy())

    return np.array(kept_X), pd.DataFrame(stats), {"acc_rate": accepts / float(N), "elapsed": time.time() - t0}


def run_chain_generic_rmala(rng, target, N=15000, burn=3000, thin=5, h=6e-3, X_init=None, store_mats=True):
    d = target.d
    X = spd_project(np.eye(d) + 0.20 * rng.normal(size=(d, d))) if X_init is None else spd_project(X_init)
    sigma2_scale = 2.0 * h
    accepts, stats, kept_X = 0, [], []
    t0 = time.time()

    for k in range(N):
        logpi_x = target.logpi_leb(X)
        x = sym_to_orthovec(X)
        gradx = sym_to_orthovec(target.grad_E_logpi_leb(X))
        Gx = metric_matrix_sym_affine_invariant(X)
        Gx_inv = np.linalg.inv(Gx)
        mean_x = x + h * (Gx_inv @ gradx)
        y = rng.multivariate_normal(mean=mean_x, cov=sigma2_scale * Gx_inv)
        Y = orthovec_to_sym(y, d)

        if is_spd(Y):
            logpi_y = target.logpi_leb(Y)
            grady = sym_to_orthovec(target.grad_E_logpi_leb(Y))
            Gy = metric_matrix_sym_affine_invariant(Y)
            Gy_inv = np.linalg.inv(Gy)
            mean_y = y + h * (Gy_inv @ grady)
            logq_xy = log_gauss_vec_metric(y, mean_x, Gx, sigma2_scale)
            logq_yx = log_gauss_vec_metric(x, mean_y, Gy, sigma2_scale)
            logr = (logpi_y + logq_yx) - (logpi_x + logq_xy)
            if np.log(rng.uniform()) < min(0.0, logr):
                X = spd_project(Y)
                accepts += 1

        if k >= burn and ((k - burn) % thin == 0):
            stats.append(record_stats(X, target))
            if store_mats:
                kept_X.append(X.copy())

    return np.array(kept_X), pd.DataFrame(stats), {"acc_rate": accepts / float(N), "elapsed": time.time() - t0}


def pdhmc_single_proposal(rng, X, target, eps, L):
    X0 = X.copy()
    X_curr = X.copy()
    Xs, _ = spd_sqrt_and_invsqrt(X_curr)
    d = X.shape[0]
    S = random_sym_normal(d, rng)
    V = sym(Xs @ S @ Xs)
    H0 = target.Phi(X_curr) + 0.5 * fro_norm2(S)
    V = sym(V - 0.5 * eps * target.grad_g_Phi_U(X_curr))

    for ell in range(L):
        Xs_curr, Xis_curr = spd_sqrt_and_invsqrt(X_curr)
        S_free = sym(Xis_curr @ V @ Xis_curr)
        X_next = ai_exp_from_congruence(X_curr, eps * S_free)
        Xs_next, _ = spd_sqrt_and_invsqrt(X_next)
        V = sym(Xs_next @ S_free @ Xs_next)
        X_curr = X_next
        grad = target.grad_g_Phi_U(X_curr)
        V = sym(V - (0.5 * eps if ell == L - 1 else eps) * grad)

    V = -V
    _, Xis_end = spd_sqrt_and_invsqrt(X_curr)
    S_end = sym(Xis_end @ V @ Xis_end)
    H1 = target.Phi(X_curr) + 0.5 * fro_norm2(S_end)
    logr = -(H1 - H0)
    if np.log(rng.uniform()) < min(0.0, logr):
        return X_curr, True
    return X0, False


def run_chain_pdhmc_like(rng, target, N=12000, burn=2000, thin=5, eps=5e-2, L=6, X_init=None, store_mats=True):
    d = target.d
    X = spd_project(np.eye(d) + 0.20 * rng.normal(size=(d, d))) if X_init is None else spd_project(X_init)
    accepts, stats, kept_X = 0, [], []
    t0 = time.time()
    for k in range(N):
        X, acc = pdhmc_single_proposal(rng, X, target, eps=eps, L=L)
        accepts += int(acc)
        if k >= burn and ((k - burn) % thin == 0):
            stats.append(record_stats(X, target))
            if store_mats:
                kept_X.append(X.copy())
    return np.array(kept_X), pd.DataFrame(stats), {"acc_rate": accepts / float(N), "elapsed": time.time() - t0}


def split_rhat(chains_1d):
    split = []
    for c in chains_1d:
        T = len(c)
        h = T // 2
        split.append(c[:h])
        split.append(c[h:2 * h])
    split = np.array(split)
    _, n = split.shape
    means = split.mean(axis=1)
    vars_ = split.var(axis=1, ddof=1)
    W = vars_.mean()
    B = n * means.var(ddof=1)
    var_hat = (n - 1) / n * W + (1 / n) * B
    return float(np.sqrt(var_hat / W))


def ess_1d(x):
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
        acf[lag] = np.dot(x[:n - lag], x[lag:]) / (n - lag) / var
    tau = 1.0
    for k in range(1, max_lag, 2):
        s = acf[k] + acf[k + 1]
        if s <= 0:
            break
        tau += 2 * s
    return float(max(1.0, min(n, n / tau)))


def build_default_true_matrix(d: int) -> np.ndarray:
    X = np.eye(d)
    for i in range(d):
        X[i, i] = 1.4 - 0.2 * i if i < 4 else 1.0
    for i in range(d):
        for j in range(i + 1, d):
            X[i, j] = X[j, i] = 0.25 / (abs(i - j) + 1)
    return spd_project(X)


def main():
    parser = argparse.ArgumentParser(description="Experiment 1: intrinsic SPD posterior, general d.")
    parser.add_argument("--d", type=int, default=3)
    parser.add_argument("--n_data", type=int, default=10)
    parser.add_argument("--sigma_data", type=float, default=0.35)
    parser.add_argument("--n_chains", type=int, default=4)
    parser.add_argument("--N", type=int, default=15000)
    parser.add_argument("--burn", type=int, default=3000)
    parser.add_argument("--thin", type=int, default=5)
    parser.add_argument("--h_cone", type=float, default=8e-3)
    parser.add_argument("--h_euclid", type=float, default=8e-4)
    parser.add_argument("--h_rmala", type=float, default=6e-3)
    parser.add_argument("--pdhmc_N", type=int, default=12000)
    parser.add_argument("--pdhmc_burn", type=int, default=2000)
    parser.add_argument("--pdhmc_eps", type=float, default=5e-2)
    parser.add_argument("--pdhmc_L", type=int, default=6)
    parser.add_argument("--lambda_", type=float, default=4.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--kappa", type=float, default=20.0)
    parser.add_argument("--seed_data", type=int, default=2026)
    parser.add_argument("--outdir", type=str, default="")
    args = parser.parse_args()

    outdir = args.outdir or f"experiment1_spd_d{args.d}_{time.strftime('%Y%m%d_%H%M%S')}"
    Path(outdir, "results").mkdir(parents=True, exist_ok=True)
    Path(outdir, "plots").mkdir(parents=True, exist_ok=True)

    rng0 = np.random.default_rng(args.seed_data)
    X_true = build_default_true_matrix(args.d)
    data = [random_spd_around(rng0, X_true, sigma=args.sigma_data) for _ in range(args.n_data)]
    target = TargetSPD_CovData(data, lambda_=args.lambda_, beta=args.beta, kappa=args.kappa)

    np.save(Path(outdir, "results", "X_true.npy"), X_true)
    np.save(Path(outdir, "results", "Xbar_ai_mean.npy"), target.Xbar)
    for i, S in enumerate(data):
        np.save(Path(outdir, "results", f"S_data_{i}.npy"), S)

    methods = {
        "cone_geom_mala": (run_chain_cone_geom_mala, {"N": args.N, "burn": args.burn, "thin": args.thin, "h": args.h_cone}),
        "Euclidean_MALA": (run_chain_euclidean_mala, {"N": args.N, "burn": args.burn, "thin": args.thin, "h": args.h_euclid}),
        "generic_RMALA": (run_chain_generic_rmala, {"N": args.N, "burn": args.burn, "thin": args.thin, "h": args.h_rmala}),
        "PDHMC_like": (run_chain_pdhmc_like, {"N": args.pdhmc_N, "burn": args.pdhmc_burn, "thin": args.thin, "eps": args.pdhmc_eps, "L": args.pdhmc_L}),
    }

    chains_stats = {m: [] for m in methods}
    chains_meta = {m: [] for m in methods}

    seeds = list(range(args.n_chains))
    for mname, (runner, kwargs) in methods.items():
        for c in range(args.n_chains):
            rng = np.random.default_rng(seeds[c])
            X_init = spd_project(np.eye(args.d) + 0.20 * rng.normal(size=(args.d, args.d)))
            mats, df, meta = runner(rng, target, X_init=X_init, store_mats=True, **kwargs)
            chains_stats[mname].append(df)
            chains_meta[mname].append(meta)
            df.to_csv(Path(outdir, "results", f"{mname}_chain{c}_stats.csv"), index=False)
            np.save(Path(outdir, "results", f"{mname}_chain{c}_samples.npy"), mats)

    obs = ["logdet", "lmin", "d2_bar", "tr", "Phi"]
    rows = []
    for mname in methods:
        metas = chains_meta[mname]
        dfs = chains_stats[mname]
        acc = np.array([m["acc_rate"] for m in metas])
        rt = np.array([m["elapsed"] for m in metas])
        row = {
            "Method": mname,
            "Runtime_sec_per_chain": float(rt.mean()),
            "Acc_mean": float(acc.mean()),
            "Acc_sd": float(acc.std(ddof=1)),
        }
        for nm in obs:
            series = [df[nm].values for df in dfs]
            row[f"Rhat_{nm}"] = split_rhat(series)
            row[f"ESSsec_{nm}"] = float(np.mean([ess_1d(s) / max(1e-12, m["elapsed"]) for s, m in zip(series, metas)]))
        rows.append(row)
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(Path(outdir, "results", "summary.csv"), index=False)

    pooled_df = compute_pooled_summary(chains_stats, obs)
    pooled_df.to_csv(Path(outdir, "results", "pooled_method_summary_stats.csv"), index=False)

    for nm in obs:
        save_method_hist_overlay(chains_stats, outdir, nm)
        save_method_ecdf_overlay(chains_stats, outdir, nm)
    save_two_panel_ecdf(
        chains_stats, outdir,
        "logdet", "Phi",
        filename=f"ecdf_logdet_phi_two_panel_d{args.d}",
        title_left="(a) Log-determinant observable",
        title_right="(b) Posterior energy",
    )
    for method, dfs in chains_stats.items():
        save_chain_trace(dfs[0], outdir, method, "Phi", chain_id=0)

    print(f"Saved Experiment 1 outputs to: {outdir}")


if __name__ == "__main__":
    main()

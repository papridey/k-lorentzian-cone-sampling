from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.linalg import slogdet, inv, eigvalsh
from scipy.linalg import cholesky


# ============================================================
# Self-contained Cone-MALA / comparison experiment
# d=5, m in {20,50,100} by default.
# Minimal structural changes from the uploaded script:
#   1. Removed dependency on plotting.py.
#   2. Added m-sweep driver.
#   3. Added safer default step sizes for d=5.
#   4. Made PDHMC optional because finite-difference HMC is very slow for m=50,100.
# ============================================================


def sym(A):
    return 0.5 * (A + A.T)


def is_spd(A, tol=1e-10):
    try:
        np.linalg.cholesky(sym(A) + tol * np.eye(A.shape[0]))
        return True
    except np.linalg.LinAlgError:
        return False


def project_spd(A, eps=1e-8):
    A = sym(A)
    vals, vecs = np.linalg.eigh(A)
    vals = np.maximum(vals, eps)
    return sym(vecs @ np.diag(vals) @ vecs.T)


def mat_sqrt_inv_sqrt(X):
    X = sym(X)
    vals, vecs = np.linalg.eigh(X)
    vals = np.maximum(vals, 1e-12)
    Xsqrt = vecs @ np.diag(np.sqrt(vals)) @ vecs.T
    Xisqrt = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
    return sym(Xsqrt), sym(Xisqrt)


def logdet_spd(X):
    X = sym(X)
    sign, ld = slogdet(X)
    if sign <= 0:
        return -np.inf
    return float(ld)


def expm_sym(S):
    S = sym(S)
    vals, vecs = np.linalg.eigh(S)
    return sym(vecs @ np.diag(np.exp(vals)) @ vecs.T)


def logm_spd(A):
    A = project_spd(A)
    vals, vecs = np.linalg.eigh(A)
    vals = np.maximum(vals, 1e-15)
    return sym(vecs @ np.diag(np.log(vals)) @ vecs.T)


def spd_congruence_log(X, Y):
    _, Xisqrt = mat_sqrt_inv_sqrt(X)
    return logm_spd(sym(Xisqrt @ Y @ Xisqrt))


def sym_to_orthovec(A):
    A = sym(A)
    d = A.shape[0]
    out = []
    for i in range(d):
        out.append(A[i, i])
    for i in range(d):
        for j in range(i + 1, d):
            out.append(np.sqrt(2.0) * A[i, j])
    return np.array(out, dtype=float)


def orthovec_to_sym(v, d):
    v = np.asarray(v, dtype=float).reshape(-1)
    A = np.zeros((d, d), dtype=float)
    idx = 0
    for i in range(d):
        A[i, i] = v[idx]
        idx += 1
    for i in range(d):
        for j in range(i + 1, d):
            A[i, j] = v[idx] / np.sqrt(2.0)
            A[j, i] = A[i, j]
            idx += 1
    return sym(A)


def random_sym_normal(d, rng):
    p = d * (d + 1) // 2
    return orthovec_to_sym(rng.normal(size=p), d)


def make_cycle_graph(m):
    return [(i, (i + 1) % m) for i in range(m)]


def block_laplacian(W, edges, m, d):
    L = np.zeros((m * d, m * d), dtype=float)
    for e, (i, j) in enumerate(edges):
        We = sym(W[e])
        si = slice(i * d, (i + 1) * d)
        sj = slice(j * d, (j + 1) * d)
        L[si, si] += We
        L[sj, sj] += We
        L[si, sj] -= We
        L[sj, si] -= We
    return sym(L)


def edge_gradient_from_A(A, edges, m, d):
    grads = []
    for (i, j) in edges:
        si = slice(i * d, (i + 1) * d)
        sj = slice(j * d, (j + 1) * d)
        G = A[si, si] + A[sj, sj] - A[si, sj] - A[sj, si]
        grads.append(sym(G))
    return np.array(grads)


def sample_ground_truth_W(E, d, rng, scale=0.25):
    W = np.zeros((E, d, d), dtype=float)
    for e in range(E):
        A = rng.normal(size=(d, d))
        W[e] = scale * (A @ A.T) + 0.15 * np.eye(d)
    return np.array([project_spd(W[e]) for e in range(E)])


def sample_graph_signals(Q, n_samples, rng):
    C = sym(inv(Q))
    Lc = cholesky(C, lower=True)
    Z = rng.normal(size=(Q.shape[0], n_samples))
    return Lc @ Z


class PSDGraphPosterior:
    def __init__(self, edges, m, d, Y_train, R, lambda_tr=0.05, eta_frob=0.02, alpha_logdet=2.5):
        self.edges = edges
        self.m = int(m)
        self.d = int(d)
        self.E = len(edges)
        self.Y = Y_train
        self.N = Y_train.shape[1]
        self.Syy = Y_train @ Y_train.T
        self.R = sym(R)
        self.lambda_tr = float(lambda_tr)
        self.eta_frob = float(eta_frob)
        self.alpha_logdet = float(alpha_logdet)
        self.vol_c = 0.5 * (self.d + 1.0)

    def Q(self, W):
        return block_laplacian(W, self.edges, self.m, self.d) + self.R

    def phi_and_grad(self, W):
        W = np.array([sym(W[e]) for e in range(len(W))])
        Q = self.Q(W)
        ldQ = logdet_spd(Q)
        if not np.isfinite(ldQ):
            return np.inf, np.zeros_like(W)

        Qinv = sym(inv(Q))
        phi_lik = -0.5 * self.N * ldQ + 0.5 * np.trace(Q @ self.Syy)
        A = -0.5 * self.N * Qinv + 0.5 * self.Syy
        gradW = edge_gradient_from_A(A, self.edges, self.m, self.d)

        phi_prior = 0.0
        for e in range(self.E):
            We = sym(W[e])
            ldWe = logdet_spd(We)
            if not np.isfinite(ldWe):
                return np.inf, np.zeros_like(W)
            Winv = sym(inv(We))
            phi_prior += self.lambda_tr * np.trace(We)
            phi_prior += 0.5 * self.eta_frob * np.sum(We * We)
            phi_prior += -self.alpha_logdet * ldWe
            gradW[e] += self.lambda_tr * np.eye(self.d)
            gradW[e] += self.eta_frob * We
            gradW[e] += -self.alpha_logdet * Winv

        return float(phi_lik + phi_prior), np.array([sym(G) for G in gradW])

    def log_vol_density_dx(self, W):
        out = 0.0
        for e in range(self.E):
            ld = logdet_spd(W[e])
            if not np.isfinite(ld):
                return -np.inf
            out += -self.vol_c * ld
        return float(out)

    def log_target_dx(self, W):
        phi, _ = self.phi_and_grad(W)
        return -phi + self.log_vol_density_dx(W)


def log_gaussian_orthosym(S, M, var):
    s = sym_to_orthovec(S)
    m = sym_to_orthovec(M)
    p = len(s)
    r = s - m
    return float(-0.5 * p * np.log(2.0 * np.pi * var) - 0.5 * np.dot(r, r) / var)


def log_spd_exp_jacobian(S):
    vals = eigvalsh(sym(S))
    out = 0.0
    d = len(vals)
    for i in range(d):
        for j in range(i + 1, d):
            a = 0.5 * (vals[i] - vals[j])
            if abs(a) < 1e-8:
                out += (a * a) / 6.0
            else:
                out += np.log(abs(np.sinh(a) / a))
    return float(out)


# -----------------------------
# Samplers
# -----------------------------

def cone_geom_mala_fast(posterior, W0, n_steps=3000, h=8e-4, seed=0):
    rng = np.random.default_rng(seed)
    W = np.array([project_spd(W0[e]) for e in range(len(W0))])
    E, d, _ = W.shape
    samples, accepts = [], 0
    t0 = time.time()
    phi, grad = posterior.phi_and_grad(W)

    for _ in range(n_steps):
        Y = np.zeros_like(W)
        S_list, M_list = [], []
        for e in range(E):
            We = sym(W[e])
            Wsqrt, _ = mat_sqrt_inv_sqrt(We)
            M = -h * sym(Wsqrt @ grad[e] @ Wsqrt)
            Z = random_sym_normal(d, rng)
            S = sym(M + np.sqrt(2.0 * h) * Z)
            Ye = Wsqrt @ expm_sym(S) @ Wsqrt
            Y[e] = project_spd(Ye)
            S_list.append(S)
            M_list.append(M)

        phiY, gradY = posterior.phi_and_grad(Y)
        if not np.isfinite(phiY):
            samples.append(W.copy())
            continue

        logq_fwd = 0.0
        logq_rev = 0.0
        for e in range(E):
            We = sym(W[e])
            Ye = sym(Y[e])
            S = S_list[e]
            Mx = M_list[e]
            logq_fwd += log_gaussian_orthosym(S, Mx, 2.0 * h) - log_spd_exp_jacobian(S)

            T = spd_congruence_log(Ye, We)
            Ysqrt, _ = mat_sqrt_inv_sqrt(Ye)
            My = -h * sym(Ysqrt @ gradY[e] @ Ysqrt)
            logq_rev += log_gaussian_orthosym(T, My, 2.0 * h) - log_spd_exp_jacobian(T)

        log_alpha = (-phiY + logq_rev) - (-phi + logq_fwd)
        if np.log(rng.uniform()) < min(0.0, log_alpha):
            W = Y
            phi = phiY
            grad = gradY
            accepts += 1
        samples.append(W.copy())

    return np.array(samples), accepts / float(n_steps), time.time() - t0


def euclidean_mala(posterior, W0, n_steps=3000, h=2e-5, seed=1):
    rng = np.random.default_rng(seed)
    W = np.array([project_spd(W0[e]) for e in range(len(W0))])
    E, d, _ = W.shape
    samples, accepts = [], 0
    t0 = time.time()
    phi, grad = posterior.phi_and_grad(W)
    logpi = posterior.log_target_dx(W)

    for _ in range(n_steps):
        Y = np.zeros_like(W)
        for e in range(E):
            x = sym_to_orthovec(W[e])
            g_dx_mat = grad[e] + posterior.vol_c * inv(W[e])
            g_dx = sym_to_orthovec(g_dx_mat)
            mu = x - h * g_dx
            y = mu + np.sqrt(2.0 * h) * rng.normal(size=len(x))
            Y[e] = orthovec_to_sym(y, d)

        if not all(is_spd(Y[e]) for e in range(E)):
            samples.append(W.copy())
            continue

        Y = np.array([project_spd(Y[e]) for e in range(E)])
        phiY, gradY = posterior.phi_and_grad(Y)
        logpiY = posterior.log_target_dx(Y)
        if not np.isfinite(logpiY):
            samples.append(W.copy())
            continue

        logq_fwd = 0.0
        logq_rev = 0.0
        for e in range(E):
            x = sym_to_orthovec(W[e])
            y = sym_to_orthovec(Y[e])
            g_dx_x = sym_to_orthovec(grad[e] + posterior.vol_c * inv(W[e]))
            g_dx_y = sym_to_orthovec(gradY[e] + posterior.vol_c * inv(Y[e]))
            mu_x = x - h * g_dx_x
            mu_y = y - h * g_dx_y
            p = len(x)
            logq_fwd += -0.5 * p * np.log(4.0 * np.pi * h) - np.sum((y - mu_x) ** 2) / (4.0 * h)
            logq_rev += -0.5 * p * np.log(4.0 * np.pi * h) - np.sum((x - mu_y) ** 2) / (4.0 * h)

        log_alpha = (logpiY + logq_rev) - (logpi + logq_fwd)
        if np.log(rng.uniform()) < min(0.0, log_alpha):
            W = Y
            phi = phiY
            grad = gradY
            logpi = logpiY
            accepts += 1
        samples.append(W.copy())

    return np.array(samples), accepts / float(n_steps), time.time() - t0


def generic_rmala(posterior, W0, n_steps=3000, h=3e-5, seed=2):
    rng = np.random.default_rng(seed)
    W = np.array([project_spd(W0[e]) for e in range(len(W0))])
    E, d, _ = W.shape
    samples, accepts = [], 0
    t0 = time.time()
    phi, grad = posterior.phi_and_grad(W)
    logpi = posterior.log_target_dx(W)
    c_jac = 0.5 * (d + 1.0)

    for _ in range(n_steps):
        Y = np.zeros_like(W)
        for e in range(E):
            We = sym(W[e])
            Wsqrt, _ = mat_sqrt_inv_sqrt(We)
            g_dx_mat = grad[e] + posterior.vol_c * inv(We)
            M = -h * sym(Wsqrt @ g_dx_mat @ Wsqrt)
            Z = random_sym_normal(d, rng)
            S = sym(M + np.sqrt(2.0 * h) * Z)
            U = Wsqrt @ S @ Wsqrt
            Y[e] = sym(We + U)

        if not all(is_spd(Y[e]) for e in range(E)):
            samples.append(W.copy())
            continue

        Y = np.array([project_spd(Y[e]) for e in range(E)])
        phiY, gradY = posterior.phi_and_grad(Y)
        logpiY = posterior.log_target_dx(Y)
        if not np.isfinite(logpiY):
            samples.append(W.copy())
            continue

        logq_fwd = 0.0
        logq_rev = 0.0
        for e in range(E):
            We = sym(W[e])
            Ye = sym(Y[e])
            Wsqrt, Wisqrt = mat_sqrt_inv_sqrt(We)
            Ysqrt, Yisqrt = mat_sqrt_inv_sqrt(Ye)
            Sf = sym(Wisqrt @ (Ye - We) @ Wisqrt)
            Sr = sym(Yisqrt @ (We - Ye) @ Yisqrt)
            g_dx_x = grad[e] + posterior.vol_c * inv(We)
            g_dx_y = gradY[e] + posterior.vol_c * inv(Ye)
            Mx = -h * sym(Wsqrt @ g_dx_x @ Wsqrt)
            My = -h * sym(Ysqrt @ g_dx_y @ Ysqrt)
            logq_fwd += log_gaussian_orthosym(Sf, Mx, 2.0 * h) - c_jac * logdet_spd(We)
            logq_rev += log_gaussian_orthosym(Sr, My, 2.0 * h) - c_jac * logdet_spd(Ye)

        log_alpha = (logpiY + logq_rev) - (logpi + logq_fwd)
        if np.log(rng.uniform()) < min(0.0, log_alpha):
            W = Y
            phi = phiY
            grad = gradY
            logpi = logpiY
            accepts += 1
        samples.append(W.copy())

    return np.array(samples), accepts / float(n_steps), time.time() - t0


# -----------------------------
# Optional PDHMC-like baseline
# Warning: finite-difference gradient is very expensive for d=5, m=50/100.
# -----------------------------

def W_to_Z(W):
    return np.array([logm_spd(W[e]) for e in range(W.shape[0])])


def Z_to_W(Z):
    return np.array([project_spd(expm_sym(Z[e])) for e in range(Z.shape[0])])


def flatten_Z(Z):
    return np.concatenate([sym_to_orthovec(Z[e]) for e in range(Z.shape[0])])


def unflatten_Z(z, E, d):
    p = d * (d + 1) // 2
    Z = np.zeros((E, d, d), dtype=float)
    for e in range(E):
        Z[e] = orthovec_to_sym(z[e * p:(e + 1) * p], d)
    return Z


def log_jac_matrix_exp(Zblock):
    z = eigvalsh(sym(Zblock))
    out = np.sum(z)
    d = len(z)
    for i in range(d):
        for j in range(i + 1, d):
            diff = z[i] - z[j]
            if abs(diff) < 1e-8:
                out += z[i]
            else:
                num = np.exp(z[i]) - np.exp(z[j])
                out += np.log(abs(num / diff))
    return float(out)


def log_target_Z(posterior, zvec):
    E, d = posterior.E, posterior.d
    Z = unflatten_Z(zvec, E, d)
    W = Z_to_W(Z)
    logpi_dx = posterior.log_target_dx(W)
    if not np.isfinite(logpi_dx):
        return -np.inf
    return float(logpi_dx + sum(log_jac_matrix_exp(Z[e]) for e in range(E)))


def finite_diff_grad(f, x, eps=1e-5):
    g = np.zeros_like(x)
    for i in range(len(x)):
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        fp, fm = f(xp), f(xm)
        g[i] = (fp - fm) / (2.0 * eps) if np.isfinite(fp) and np.isfinite(fm) else 0.0
    return g


def pdhmc_like(posterior, W0, n_steps=500, eps=2e-2, L=4, seed=3):
    rng = np.random.default_rng(seed)
    z = flatten_Z(W_to_Z(W0))
    dim = len(z)
    f = lambda zz: log_target_Z(posterior, zz)
    logp = f(z)
    samples, accepts = [], 0
    t0 = time.time()

    for _ in range(n_steps):
        p = rng.normal(size=dim)
        z_old = z.copy()
        p_old = p.copy()
        logp_old = logp

        g = finite_diff_grad(f, z, eps=1e-5)
        p = p + 0.5 * eps * g
        z_new = z.copy()
        for ell in range(L):
            z_new = z_new + eps * p
            g_new = finite_diff_grad(f, z_new, eps=1e-5)
            if ell != L - 1:
                p = p + eps * g_new
        p = -(p + 0.5 * eps * g_new)
        logp_new = f(z_new)

        H_old = -logp_old + 0.5 * np.dot(p_old, p_old)
        H_new = -logp_new + 0.5 * np.dot(p, p)
        if np.isfinite(logp_new) and np.log(rng.uniform()) < min(0.0, -(H_new - H_old)):
            z = z_new
            logp = logp_new
            accepts += 1
        else:
            z = z_old
            logp = logp_old

        samples.append(Z_to_W(unflatten_Z(z, posterior.E, posterior.d)).copy())

    return np.array(samples), accepts / float(n_steps), time.time() - t0


# -----------------------------
# Diagnostics and plotting
# -----------------------------

def ess_1d(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 10:
        return float(n)
    x = x - np.mean(x)
    var = np.dot(x, x) / n
    if var <= 1e-14:
        return float(n)
    max_lag = min(n // 2, 1000)
    rho_sum = 0.0
    for lag in range(1, max_lag):
        acov = np.dot(x[:-lag], x[lag:]) / (n - lag)
        rho = acov / var
        if rho <= 0:
            break
        rho_sum += rho
    return float(n / (1.0 + 2.0 * rho_sum))


def split_rhat(chains_1d):
    chains = [np.asarray(c, dtype=float) for c in chains_1d]
    min_len = min(len(c) for c in chains)
    min_len = 2 * (min_len // 2)
    if min_len < 20 or len(chains) < 2:
        return np.nan
    chains = np.array([c[:min_len] for c in chains])
    split = []
    for c in chains:
        h = min_len // 2
        split.append(c[:h])
        split.append(c[h:])
    split = np.array(split)
    _, n = split.shape
    chain_means = split.mean(axis=1)
    chain_vars = split.var(axis=1, ddof=1)
    W = chain_vars.mean()
    B = n * chain_means.var(ddof=1)
    var_hat = ((n - 1) / n) * W + B / n
    return np.nan if W <= 0 else float(np.sqrt(var_hat / W))


def observables(W_samples, posterior, W_star=None, Y_test=None):
    out = {"logdet_Q": [], "lambda_min_Q": [], "Phi": [], "rel_W_error": [], "test_nll": []}
    for W in W_samples:
        Q = posterior.Q(W)
        phi, _ = posterior.phi_and_grad(W)
        out["logdet_Q"].append(logdet_spd(Q))
        out["lambda_min_Q"].append(float(np.min(eigvalsh(Q))))
        out["Phi"].append(float(phi))
        if W_star is not None:
            num = np.sqrt(np.sum((W - W_star) ** 2))
            den = np.sqrt(np.sum(W_star ** 2))
            out["rel_W_error"].append(float(num / den))
        if Y_test is not None:
            Nt = Y_test.shape[1]
            S_test = Y_test @ Y_test.T
            nll = -0.5 * Nt * logdet_spd(Q) + 0.5 * np.trace(Q @ S_test)
            out["test_nll"].append(float(nll / Nt))
    return {k: np.array(v) for k, v in out.items() if len(v) > 0}


def ecdf_values(x):
    x = np.sort(np.asarray(x, dtype=float))
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def save_two_panel_ecdf(plot_dfs, outdir, x_left, x_right, filename, title_left, title_right):
    import matplotlib.pyplot as plt

    plot_dir = Path(outdir, "plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for method_name, dfs in plot_dfs.items():
        vals_left = np.concatenate([df[x_left].dropna().to_numpy() for df in dfs])
        vals_right = np.concatenate([df[x_right].dropna().to_numpy() for df in dfs])
        xl, yl = ecdf_values(vals_left)
        xr, yr = ecdf_values(vals_right)
        axes[0].plot(xl, yl, linewidth=2, label=method_name)
        axes[1].plot(xr, yr, linewidth=2, label=method_name)

    axes[0].set_title(title_left)
    axes[1].set_title(title_right)
    axes[0].set_xlabel(x_left)
    axes[1].set_xlabel(x_right)
    axes[0].set_ylabel("ECDF")
    axes[1].set_ylabel("ECDF")
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)

    pdf_path = plot_dir / f"{filename}.pdf"
    png_path = plot_dir / f"{filename}.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_accept_runtime_plot(summary_df, outdir, m):
    import matplotlib.pyplot as plt

    plot_dir = Path(outdir, "plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    df = summary_df.sort_values("Method")
    ax.bar(df["Method"], df["accept_mean"])
    ax.set_ylabel("Mean acceptance rate")
    ax.set_title(f"Acceptance rates, d=5, m={m}")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(plot_dir / f"acceptance_d5_m{m}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(plot_dir / f"acceptance_d5_m{m}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_initial_W(E, d, chain_index):
    rng_init = np.random.default_rng(1000 + chain_index)
    return np.array([
        project_spd(0.35 * np.eye(d) + 0.03 * sym(rng_init.normal(size=(d, d))))
        for _ in range(E)
    ])


def default_steps_for_m(m, base_steps):
    # Keeps m=100 feasible on a laptop unless user overrides --n_steps.
    if base_steps is not None:
        return base_steps
    if m <= 20:
        return 6000
    if m <= 50:
        return 4000
    return 2500


def run_one_experiment(args, m):
    d = args.d
    n_steps = default_steps_for_m(m, args.n_steps)
    burn = min(args.burn, max(0, n_steps // 3))

    outdir = Path(args.outdir) / f"graph_posterior_d{d}_m{m}"
    Path(outdir, "results").mkdir(parents=True, exist_ok=True)
    Path(outdir, "plots").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed + 17 * m)
    edges = make_cycle_graph(m)
    E = len(edges)

    W_star = sample_ground_truth_W(E, d, rng, scale=args.truth_scale)
    R = args.R_scale * np.eye(m * d)
    Q_star = block_laplacian(W_star, edges, m, d) + R
    Y_train = sample_graph_signals(Q_star, args.n_train, rng)
    Y_test = sample_graph_signals(Q_star, args.n_test, rng)

    posterior = PSDGraphPosterior(
        edges=edges,
        m=m,
        d=d,
        Y_train=Y_train,
        R=R,
        lambda_tr=args.lambda_tr,
        eta_frob=args.eta_frob,
        alpha_logdet=args.alpha_logdet,
    )

    np.save(Path(outdir, "results", "W_star.npy"), W_star)
    np.save(Path(outdir, "results", "Q_star.npy"), Q_star)
    np.save(Path(outdir, "results", "Y_train.npy"), Y_train)
    np.save(Path(outdir, "results", "Y_test.npy"), Y_test)

    method_specs = {
        "cone_geom_mala": {
            "runner": cone_geom_mala_fast,
            "kwargs": {"n_steps": n_steps, "h": args.h_cone},
            "burn": burn,
        },
        "Euclidean_MALA": {
            "runner": euclidean_mala,
            "kwargs": {"n_steps": n_steps, "h": args.h_euclid},
            "burn": burn,
        },
        "generic_RMALA": {
            "runner": generic_rmala,
            "kwargs": {"n_steps": n_steps, "h": args.h_rmala},
            "burn": burn,
        },
    }

    if args.include_pdhmc:
        method_specs["PDHMC_like"] = {
            "runner": pdhmc_like,
            "kwargs": {"n_steps": args.pdhmc_steps, "eps": args.pdhmc_eps, "L": args.pdhmc_L},
            "burn": min(args.pdhmc_burn, max(0, args.pdhmc_steps // 3)),
        }

    config = vars(args).copy()
    config.update({"active_m": m, "active_d": d, "active_n_steps": n_steps, "active_burn": burn})
    with open(Path(outdir, "results", "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    rows = []
    plot_dfs = {}
    seeds = [10_000 + 100 * m + i for i in range(args.n_chains)]

    print(f"\n=== Running d={d}, m={m}, E={E}, n_steps={n_steps}, burn={burn} ===")
    print(f"Output directory: {outdir}")

    for method_name, spec in method_specs.items():
        print(f"\nMethod: {method_name}")
        obs_chains, runtimes, accepts = [], [], []

        for c in range(args.n_chains):
            W0_chain = make_initial_W(E, d, c)
            samples, acc, rt = spec["runner"](posterior, W0_chain, seed=seeds[c], **spec["kwargs"])
            post = samples[spec["burn"]:]
            obs = observables(post, posterior, W_star=W_star, Y_test=Y_test)
            obs_chains.append(obs)
            runtimes.append(rt)
            accepts.append(acc)

            pd.DataFrame(obs).to_csv(Path(outdir, "results", f"{method_name}_chain{c}_observables.csv"), index=False)
            if args.save_samples:
                np.save(Path(outdir, "results", f"{method_name}_chain{c}_samples.npy"), samples)

            print(f"  chain {c}: accept={acc:.3f}, runtime={rt:.1f}s, kept={len(post)}")

        row = {
            "d": d,
            "m": m,
            "E": E,
            "n_steps": int(spec["kwargs"].get("n_steps", n_steps)),
            "burn": int(spec["burn"]),
            "Method": method_name,
            "runtime_mean": float(np.mean(runtimes)),
            "runtime_sd": float(np.std(runtimes, ddof=1)) if len(runtimes) > 1 else 0.0,
            "accept_mean": float(np.mean(accepts)),
            "accept_sd": float(np.std(accepts, ddof=1)) if len(accepts) > 1 else 0.0,
        }

        for key in ["rel_W_error", "test_nll", "logdet_Q", "lambda_min_Q", "Phi"]:
            chains = [obs[key] for obs in obs_chains if key in obs]
            pooled = np.concatenate(chains)
            row[f"{key}_mean"] = float(np.mean(pooled))
            row[f"{key}_sd"] = float(np.std(pooled, ddof=1))
            row[f"{key}_Rhat"] = split_rhat(chains)
            row[f"{key}_ESSsec"] = float(sum(ess_1d(cn) for cn in chains) / max(1e-12, np.sum(runtimes)))

        rows.append(row)

        plot_dfs[method_name] = [
            pd.DataFrame({
                "logdet_Q": obs["logdet_Q"],
                "Phi": obs["Phi"],
                "lambda_min_Q": obs["lambda_min_Q"],
                "rel_W_error": obs["rel_W_error"],
                "test_nll": obs["test_nll"],
            })
            for obs in obs_chains
        ]

    summary_df = pd.DataFrame(rows)

    # Main summary file inside the experiment-specific results folder.
    summary_path = Path(outdir, "results", "summary_multichain.csv")
    summary_df.to_csv(summary_path, index=False)

    # Reviewer-friendly stable filenames.
    # These are useful when reproducing the tables in the paper.
    stable_summary_name = f"summary_d{d}_m{m}.csv"
    stable_summary_path = Path(outdir, "results", stable_summary_name)
    summary_df.to_csv(stable_summary_path, index=False)

    # Optional duplicate at the top-level output folder for easy collection.
    top_level_summary_path = Path(args.outdir, stable_summary_name)
    summary_df.to_csv(top_level_summary_path, index=False)

    print(f"Saved summary: {summary_path}")
    print(f"Saved reviewer-friendly summary: {stable_summary_path}")
    print(f"Saved top-level reviewer-friendly summary: {top_level_summary_path}")

    save_two_panel_ecdf(
        plot_dfs,
        outdir,
        "logdet_Q",
        "Phi",
        filename=f"ecdf_logdet_phi_two_panel_d{d}_m{m}",
        title_left="(a) Log-determinant observable",
        title_right="(b) Posterior energy",
    )
    save_two_panel_ecdf(
        plot_dfs,
        outdir,
        "rel_W_error",
        "test_nll",
        filename=f"ecdf_error_testnll_two_panel_d{d}_m{m}",
        title_left="(a) Relative edge-weight error",
        title_right="(b) Test negative log-likelihood",
    )
    save_accept_runtime_plot(summary_df, outdir, m)

    print(f"Saved summary: {summary_path}")
    return summary_df


def parse_m_list(s):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Cone-MALA graph posterior sweep for d=5 and m={20,50,100}.")

    # Main sweep settings
    parser.add_argument("--d", type=int, default=5)
    parser.add_argument("--m_list", type=str, default="20,50,100", help="Comma-separated list, e.g. 20,50,100")
    parser.add_argument("--n_train", type=int, default=80)
    parser.add_argument("--n_test", type=int, default=80)
    parser.add_argument("--n_chains", type=int, default=4)
    parser.add_argument("--n_steps", type=int, default=None, help="If omitted: 6000 for m=20, 4000 for m=50, 2500 for m=100.")
    parser.add_argument("--burn", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--outdir", type=str, default="cone_mala_d5_m_sweep_outputs")
    parser.add_argument("--save_samples", action="store_true", help="Save full samples. This can be large for m=100.")

    # Target/prior settings
    parser.add_argument("--truth_scale", type=float, default=0.25)
    parser.add_argument("--R_scale", type=float, default=0.75)
    parser.add_argument("--lambda_tr", type=float, default=0.05)
    parser.add_argument("--eta_frob", type=float, default=0.02)
    parser.add_argument("--alpha_logdet", type=float, default=2.5)

    # Step sizes tuned conservatively for d=5 stability.
    parser.add_argument("--h_cone", type=float, default=8.0e-4)
    parser.add_argument("--h_euclid", type=float, default=2.0e-5)
    parser.add_argument("--h_rmala", type=float, default=3.0e-5)

    # Optional finite-difference HMC baseline.
    parser.add_argument("--include_pdhmc", action="store_true")
    parser.add_argument("--pdhmc_steps", type=int, default=300)
    parser.add_argument("--pdhmc_burn", type=int, default=100)
    parser.add_argument("--pdhmc_eps", type=float, default=2e-2)
    parser.add_argument("--pdhmc_L", type=int, default=4)

    args = parser.parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    all_summaries = []
    for m in parse_m_list(args.m_list):
        summary_df = run_one_experiment(args, m)
        all_summaries.append(summary_df)

    combined = pd.concat(all_summaries, ignore_index=True)
    combined_path = Path(args.outdir, "combined_summary_d5_m_sweep.csv")
    combined.to_csv(combined_path, index=False)
    print(f"\nSaved combined summary: {combined_path}")

    # Compact console view.
    cols = [
        "d", "m", "Method", "accept_mean", "runtime_mean",
        "rel_W_error_mean", "rel_W_error_ESSsec",
        "test_nll_mean", "test_nll_ESSsec",
        "Phi_Rhat",
    ]
    cols = [c for c in cols if c in combined.columns]
    print("\nCompact summary:")
    print(combined[cols].to_string(index=False))


if __name__ == "__main__":
    main()

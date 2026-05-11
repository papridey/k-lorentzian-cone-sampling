"""
Sensitivity-analysis code for pullback log-det geometry.

This script validates the local metric score

    s(U) = tr(X^{-1} Delta X^{-1} Delta)

against the finite-difference curvature of

    phi(X) = -log det(X).

It produces a two-panel figure:
    Panel A: metric score vs finite-difference curvature
    Panel B: sensitivity-mass capture under metric ranking

The setup follows the PSD-weighted graph model:
    X(W) = L(W) + R,

where each edge e carries a PSD matrix W_e in S_+^d.
Perturbations are rank-one edge directions:
    U_e = u u^T,     U_{e'} = 0 for e' != e.

Author: self-contained version for reproducible Fig. 1-style sensitivity test.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Basic linear-algebra utilities
# ============================================================

def symmetrize(A):
    """Return the symmetric part of A."""
    return 0.5 * (A + A.T)


def random_spd(d, rng, scale=1.0, ridge=1.0):
    """
    Generate a random SPD matrix of size d x d.

    Parameters
    ----------
    d : int
        Matrix dimension.
    rng : numpy.random.Generator
        Random number generator.
    scale : float
        Scale of the random component.
    ridge : float
        Positive multiple of identity added for stability.

    Returns
    -------
    A : ndarray
        SPD matrix.
    """
    Z = rng.normal(size=(d, d))
    A = Z @ Z.T
    A = scale * A / max(d, 1) + ridge * np.eye(d)
    return symmetrize(A)


def neg_logdet(X):
    """
    Compute phi(X) = -log det(X).

    Returns +inf if X is not positive definite.
    """
    X = symmetrize(X)
    sign, logdet = np.linalg.slogdet(X)
    if sign <= 0:
        return np.inf
    return -logdet


def min_eig(X):
    """Smallest eigenvalue of a symmetric matrix."""
    return np.linalg.eigvalsh(symmetrize(X))[0]


# ============================================================
# 2. Graph Laplacian construction
# ============================================================

def path_graph_edges(m):
    """
    Return edges of a path graph on m nodes.

    Example:
        m = 5 gives edges
        (0,1), (1,2), (2,3), (3,4).
    """
    return [(i, i + 1) for i in range(m - 1)]


def cycle_graph_edges(m):
    """
    Return edges of a cycle graph on m nodes.
    """
    edges = [(i, i + 1) for i in range(m - 1)]
    edges.append((m - 1, 0))
    return edges


def complete_graph_edges(m):
    """
    Return edges of the complete graph on m nodes.
    """
    return [(i, j) for i in range(m) for j in range(i + 1, m)]


def block_laplacian(edge_mats, edges, m, d):
    """
    Construct the block graph Laplacian

        L(W) = sum_e b_e b_e^T \otimes W_e,

    where b_e has +1 at one endpoint and -1 at the other.

    Parameters
    ----------
    edge_mats : list of ndarray
        List of d x d symmetric edge matrices W_e.
    edges : list of tuple
        Edge list [(i,j), ...].
    m : int
        Number of graph nodes.
    d : int
        PSD block dimension.

    Returns
    -------
    L : ndarray
        Block Laplacian of size (m*d) x (m*d).
    """
    n = m * d
    L = np.zeros((n, n))

    for W, (i, j) in zip(edge_mats, edges):
        W = symmetrize(W)

        ii = slice(i * d, (i + 1) * d)
        jj = slice(j * d, (j + 1) * d)

        L[ii, ii] += W
        L[jj, jj] += W
        L[ii, jj] -= W
        L[jj, ii] -= W

    return symmetrize(L)


def lifted_edge_perturbation(edge_index, Ue, edges, m, d):
    """
    Construct Delta_U = L(U), where only one edge has perturbation Ue.

    Parameters
    ----------
    edge_index : int
        Index of perturbed edge.
    Ue : ndarray
        d x d symmetric perturbation matrix.
    edges : list of tuple
        Edge list.
    m : int
        Number of graph nodes.
    d : int
        PSD block dimension.

    Returns
    -------
    Delta : ndarray
        Lifted perturbation matrix.
    """
    zero = np.zeros((d, d))
    edge_mats = [zero.copy() for _ in edges]
    edge_mats[edge_index] = symmetrize(Ue)
    return block_laplacian(edge_mats, edges, m, d)


# ============================================================
# 3. Metric score and finite-difference curvature
# ============================================================

def metric_score(X, Delta):
    """
    Compute the exact pullback log-det metric score

        s = tr(X^{-1} Delta X^{-1} Delta).

    This avoids explicitly forming X^{-1} when possible.
    """
    X = symmetrize(X)
    Delta = symmetrize(Delta)

    # Solve X A = Delta, so A = X^{-1} Delta.
    A = np.linalg.solve(X, Delta)

    # s = tr(X^{-1} Delta X^{-1} Delta) = tr(A A)
    # Since Delta and X are symmetric, this should be nonnegative.
    s = np.trace(A @ A)

    return float(np.real(s))


def safe_fd_epsilon(X, Delta, eps_base=1e-4, safety=0.25):
    """
    Choose a finite-difference step eps so that both

        X + eps Delta and X - eps Delta

    remain positive definite.

    A sufficient condition is based on

        || X^{-1/2} Delta X^{-1/2} ||_op.

    If lambda_max_abs is the largest absolute eigenvalue of
    X^{-1/2} Delta X^{-1/2}, then X +/- eps Delta stay SPD
    when eps * lambda_max_abs < 1.

    Parameters
    ----------
    X : ndarray
        SPD matrix.
    Delta : ndarray
        Symmetric perturbation.
    eps_base : float
        Desired finite-difference step.
    safety : float
        Safety factor below the SPD boundary.

    Returns
    -------
    eps : float
        Safe finite-difference step.
    """
    X = symmetrize(X)
    Delta = symmetrize(Delta)

    # Compute X^{-1/2} Delta X^{-1/2}.
    evals, evecs = np.linalg.eigh(X)
    if np.min(evals) <= 0:
        raise ValueError("X must be positive definite.")

    X_inv_sqrt = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    A = symmetrize(X_inv_sqrt @ Delta @ X_inv_sqrt)

    lam_abs = np.max(np.abs(np.linalg.eigvalsh(A)))

    if lam_abs <= 1e-14:
        return eps_base

    eps_max = safety / lam_abs
    eps = min(eps_base, eps_max)

    return float(eps)


def finite_difference_curvature(X, Delta, eps_base=1e-4):
    """
    Compute the centered finite-difference curvature

        delta_FD =
        [phi(X + eps Delta) - 2 phi(X) + phi(X - eps Delta)] / eps^2,

    where phi(X) = -log det(X).

    The step eps is automatically reduced if needed to preserve SPD-ness.
    """
    X = symmetrize(X)
    Delta = symmetrize(Delta)

    eps = safe_fd_epsilon(X, Delta, eps_base=eps_base)

    X_plus = symmetrize(X + eps * Delta)
    X_minus = symmetrize(X - eps * Delta)

    # Extra safety check.
    if min_eig(X_plus) <= 0 or min_eig(X_minus) <= 0:
        raise FloatingPointError("Finite-difference step failed to preserve SPD.")

    phi_plus = neg_logdet(X_plus)
    phi_0 = neg_logdet(X)
    phi_minus = neg_logdet(X_minus)

    fd = (phi_plus - 2.0 * phi_0 + phi_minus) / (eps ** 2)

    return float(fd), eps


# ============================================================
# 4. Sensitivity experiment
# ============================================================

def generate_rank_one_directions(num_dirs, d, rng, normalize=True):
    """
    Generate rank-one PSD perturbation directions U = u u^T.

    Parameters
    ----------
    num_dirs : int
        Number of directions.
    d : int
        Matrix dimension.
    rng : numpy.random.Generator
        Random generator.
    normalize : bool
        If True, normalize u to unit Euclidean norm.

    Returns
    -------
    Us : list of ndarray
        Rank-one PSD matrices.
    """
    Us = []
    for _ in range(num_dirs):
        u = rng.normal(size=d)
        if normalize:
            norm_u = np.linalg.norm(u)
            if norm_u > 0:
                u = u / norm_u
        U = np.outer(u, u)
        Us.append(symmetrize(U))
    return Us


def run_sensitivity_experiment(
    d=3,
    m=5,
    graph_type="cycle",
    dirs_per_edge=300,
    seed=123,
    eps_base=1e-4,
    R_ridge=2.0,
    W_ridge=0.5,
):
    """
    Run the sensitivity experiment for PSD edge dimension d.

    Parameters
    ----------
    d : int
        PSD block dimension. The paper's Fig. 1 uses d = 3.
    m : int
        Number of graph nodes.
    graph_type : str
        One of {"path", "cycle", "complete"}.
    dirs_per_edge : int
        Number of rank-one directions per edge.
    seed : int
        Random seed.
    eps_base : float
        Base finite-difference step.
    R_ridge : float
        Stabilizer strength for R.
    W_ridge : float
        Ridge for edge PSD weights.

    Returns
    -------
    results : dict
        Dictionary containing metric scores, finite-difference curvatures,
        selected epsilons, and metadata.
    """
    rng = np.random.default_rng(seed)

    if graph_type == "path":
        edges = path_graph_edges(m)
    elif graph_type == "cycle":
        edges = cycle_graph_edges(m)
    elif graph_type == "complete":
        edges = complete_graph_edges(m)
    else:
        raise ValueError("graph_type must be 'path', 'cycle', or 'complete'.")

    E = len(edges)

    # Random PSD edge weights W_e.
    W_edges = [
        random_spd(d, rng, scale=1.0, ridge=W_ridge)
        for _ in range(E)
    ]

    # Stabilizer R ≻ 0.
    # Use a simple isotropic SPD stabilizer to ensure X is well-conditioned.
    R = R_ridge * np.eye(m * d)

    # Lifted SPD matrix X(W) = L(W) + R.
    L = block_laplacian(W_edges, edges, m, d)
    X = symmetrize(L + R)

    if min_eig(X) <= 0:
        raise ValueError("X is not SPD. Increase R_ridge.")

    metric_scores = []
    fd_curvatures = []
    eps_used = []
    edge_ids = []

    for e_idx in range(E):
        directions = generate_rank_one_directions(dirs_per_edge, d, rng)

        for Ue in directions:
            Delta = lifted_edge_perturbation(e_idx, Ue, edges, m, d)

            s = metric_score(X, Delta)
            fd, eps = finite_difference_curvature(X, Delta, eps_base=eps_base)

            # Keep only valid positive values for log-log plotting.
            if np.isfinite(s) and np.isfinite(fd) and s > 0 and fd > 0:
                metric_scores.append(s)
                fd_curvatures.append(fd)
                eps_used.append(eps)
                edge_ids.append(e_idx)

    metric_scores = np.asarray(metric_scores)
    fd_curvatures = np.asarray(fd_curvatures)
    eps_used = np.asarray(eps_used)
    edge_ids = np.asarray(edge_ids)

    results = {
        "metric_scores": metric_scores,
        "fd_curvatures": fd_curvatures,
        "eps_used": eps_used,
        "edge_ids": edge_ids,
        "X": X,
        "L": L,
        "R": R,
        "W_edges": W_edges,
        "edges": edges,
        "d": d,
        "m": m,
        "graph_type": graph_type,
        "dirs_per_edge": dirs_per_edge,
        "seed": seed,
        "eps_base": eps_base,
    }

    return results


# ============================================================
# 5. Plotting utilities
# ============================================================

def binned_log_means(x, y, num_bins=25):
    """
    Compute binned mean and standard deviation of y over logarithmic bins in x.

    Returns
    -------
    centers : ndarray
        Geometric bin centers.
    means : ndarray
        Mean y in each bin.
    stds : ndarray
        Standard deviation of y in each bin.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    positive = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    x = x[positive]
    y = y[positive]

    log_min = np.log10(np.min(x))
    log_max = np.log10(np.max(x))
    bins = np.logspace(log_min, log_max, num_bins + 1)

    centers = []
    means = []
    stds = []

    for i in range(num_bins):
        mask = (x >= bins[i]) & (x < bins[i + 1])
        if np.sum(mask) >= 3:
            centers.append(np.sqrt(bins[i] * bins[i + 1]))
            means.append(np.mean(y[mask]))
            stds.append(np.std(y[mask]))

    return np.asarray(centers), np.asarray(means), np.asarray(stds)


def sensitivity_mass_curves(metric_scores, fd_curvatures, num_k=200):
    """
    Compute sensitivity-mass capture curves.

    The sensitivity mass is defined using FD curvature as the reference truth:

        captured mass at k =
        sum of FD scores among selected k directions / total FD score.

    Three rankings are compared:
        1. top-k by metric score,
        2. oracle top-k by FD curvature,
        3. random baseline k/N.

    Returns
    -------
    k_vals : ndarray
        Values of k.
    captured_metric : ndarray
        Captured mass using metric ranking.
    captured_oracle : ndarray
        Captured mass using oracle FD ranking.
    random_baseline : ndarray
        k/N.
    """
    s = np.asarray(metric_scores)
    fd = np.asarray(fd_curvatures)

    mask = np.isfinite(s) & np.isfinite(fd) & (s > 0) & (fd > 0)
    s = s[mask]
    fd = fd[mask]

    N = len(s)
    if N < 2:
        raise ValueError("Not enough valid directions.")

    total_mass = np.sum(fd)

    order_metric = np.argsort(s)[::-1]
    order_oracle = np.argsort(fd)[::-1]

    cum_metric = np.cumsum(fd[order_metric]) / total_mass
    cum_oracle = np.cumsum(fd[order_oracle]) / total_mass

    k_vals = np.unique(np.linspace(1, N, num_k).astype(int))

    captured_metric = cum_metric[k_vals - 1]
    captured_oracle = cum_oracle[k_vals - 1]
    random_baseline = k_vals / N

    return k_vals, captured_metric, captured_oracle, random_baseline


def make_figure_1_style_plot(
    results,
    output_dir="figures",
    filename_base="fig1_sensitivity_validation",
    dpi=600,
):
    """
    Make the two-panel Fig. 1-style plot and save as PDF and PNG.

    Parameters
    ----------
    results : dict
        Output of run_sensitivity_experiment.
    output_dir : str
        Folder for saved figures.
    filename_base : str
        Base name for files.
    dpi : int
        DPI for PNG output.
    """
    os.makedirs(output_dir, exist_ok=True)

    metric_scores = results["metric_scores"]
    fd_curvatures = results["fd_curvatures"]
    d = results["d"]

    centers, means, stds = binned_log_means(metric_scores, fd_curvatures, num_bins=25)

    k_vals, captured_metric, captured_oracle, random_baseline = sensitivity_mass_curves(
        metric_scores, fd_curvatures, num_k=250
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.0, 4.5),
        constrained_layout=True,
    )

    # ------------------------------------------------------------
    # Panel A: calibration
    # ------------------------------------------------------------
    ax = axes[0]

    ax.loglog(
        metric_scores,
        fd_curvatures,
        ".",
        alpha=0.18,
        markersize=3,
        label="directions",
    )

    ax.loglog(
        centers,
        means,
        "-",
        linewidth=2.2,
        label="binned mean of FD",
    )

    ax.fill_between(
        centers,
        np.maximum(means - stds, 1e-300),
        means + stds,
        alpha=0.18,
        label=r"$\pm 1$ std",
    )

    low = min(np.min(metric_scores), np.min(fd_curvatures))
    high = max(np.max(metric_scores), np.max(fd_curvatures))
    grid = np.logspace(np.log10(low), np.log10(high), 200)

    ax.loglog(
        grid,
        grid,
        "--",
        linewidth=1.6,
        label=r"identity: $\delta=s$",
    )

    ax.set_xlabel(
        r"metric score $s=\operatorname{tr}(X^{-1}\Delta X^{-1}\Delta)$",
        fontsize=13,
    )
    ax.set_ylabel(
        r"FD curvature $\delta_{\mathrm{FD}}$",
        fontsize=13,
    )
    ax.set_title(
        rf"Panel A. Calibration (PSD, $d={d}$)",
        fontsize=14,
    )
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=9, frameon=True)

    # ------------------------------------------------------------
    # Panel B: ranking/capture
    # ------------------------------------------------------------
    ax = axes[1]

    ax.plot(
        k_vals,
        captured_metric,
        linewidth=2.2,
        label=r"top-$k$ by $s$",
    )

    ax.plot(
        k_vals,
        captured_oracle,
        "--",
        linewidth=2.0,
        label=r"oracle: top-$k$ by FD",
    )

    ax.plot(
        k_vals,
        random_baseline,
        ":",
        linewidth=1.8,
        label=r"random baseline $k/N$",
    )

    ax.set_xlim(1, k_vals[-1])
    ax.set_ylim(0, 1.02)

    ax.set_xlabel(
        r"number of directions selected $k$",
        fontsize=13,
    )
    ax.set_ylabel(
        "captured sensitivity mass",
        fontsize=13,
    )
    ax.set_title(
        rf"Panel B. Ranking / capture (PSD, $d={d}$)",
        fontsize=14,
    )
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=9, frameon=True)

    pdf_path = os.path.join(output_dir, f"{filename_base}.pdf")
    png_path = os.path.join(output_dir, f"{filename_base}.png")

    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)

    print(f"Saved PDF: {pdf_path}")
    print(f"Saved PNG: {png_path}")

    return fig, axes


# ============================================================
# 6. Diagnostics
# ============================================================

def print_diagnostics(results):
    """
    Print useful numerical diagnostics.
    """
    s = results["metric_scores"]
    fd = results["fd_curvatures"]
    eps = results["eps_used"]
    X = results["X"]

    rel_err = np.abs(fd - s) / np.maximum(np.abs(s), 1e-14)

    corr_log = np.corrcoef(np.log(s), np.log(fd))[0, 1]

    print("\n================ Sensitivity diagnostics ================")
    print(f"d                         : {results['d']}")
    print(f"m                         : {results['m']}")
    print(f"graph type                : {results['graph_type']}")
    print(f"number of edges           : {len(results['edges'])}")
    print(f"number of directions      : {len(s)}")
    print(f"lambda_min(X)             : {min_eig(X):.6e}")
    print(f"metric score min/max      : {s.min():.6e} / {s.max():.6e}")
    print(f"FD curvature min/max      : {fd.min():.6e} / {fd.max():.6e}")
    print(f"eps used min/max          : {eps.min():.6e} / {eps.max():.6e}")
    print(f"log-log correlation       : {corr_log:.6f}")
    print(f"median relative error     : {np.median(rel_err):.6e}")
    print(f"90% relative error        : {np.quantile(rel_err, 0.90):.6e}")
    print(f"99% relative error        : {np.quantile(rel_err, 0.99):.6e}")
    print("=========================================================\n")


# ============================================================
# 7. Main script
# ============================================================

def main():
    """
    Main driver.

    You can change d, m, graph_type, dirs_per_edge, and seed below.
    For a Fig. 1-style test, use d = 3.
    """

    results = run_sensitivity_experiment(
        d=5, # 3,                  # PSD matrix dimension; Fig. 1 uses d = 3
        m=5,                  # number of graph nodes
        graph_type="cycle",   # "path", "cycle", or "complete"
        dirs_per_edge=600, #350,    # total directions = dirs_per_edge * number_of_edges
        seed=123,
        eps_base=1e-4,
        R_ridge=2.0,
        W_ridge=0.5,
    )

    print_diagnostics(results)

    make_figure_1_style_plot(
        results,
        output_dir="figures",
        filename_base="fig1_sensitivity_validation",
        dpi=600,
    )


if __name__ == "__main__":
    main()
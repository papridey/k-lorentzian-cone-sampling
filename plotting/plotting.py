
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 17,
    "axes.labelsize": 15,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "lines.linewidth": 2.2,
    "savefig.dpi": 600,
})


def ensure_plots_dir(outdir: str | Path) -> Path:
    plots_dir = Path(outdir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def pooled_values(chains_stats_list: Sequence[pd.DataFrame], col: str) -> np.ndarray:
    arrays = [df[col].to_numpy() for df in chains_stats_list if col in df.columns]
    if not arrays:
        raise ValueError(f"Column '{col}' not found in any chain dataframe.")
    return np.concatenate(arrays, axis=0)


def pretty_method_name(name: str) -> str:
    mapping = {
        "cone_geom_mala": "cone geom MALA",
        "Euclidean_MALA": "Euclidean MALA",
        "generic_RMALA": "generic RMALA",
        "PDHMC_like": "PDHMC-like",
        "fast_cone_geom_mala": "fast cone geom MALA",
    }
    return mapping.get(name, name)


def pretty_observable_label(col: str) -> str:
    mapping = {
        "logdet": r"$\log\det(X)$",
        "lmin": r"$\lambda_{\min}(X)$",
        "lmax": r"$\lambda_{\max}(X)$",
        "d2_bar": r"$d_g(X,\bar X)^2$",
        "tr": r"$\mathrm{tr}(X)$",
        "Phi": r"$\Phi(X)$",
        "logdet_Q": r"$\log\det Q(W)$",
        "lambda_min_Q": r"$\lambda_{\min}(Q(W))$",
        "rel_W_error": r"relative $W$ error",
        "test_nll": r"test NLL",
    }
    return mapping.get(col, col)


def save_figure(fig: plt.Figure, basepath: Path) -> None:
    fig.savefig(basepath.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(basepath.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def compute_pooled_summary(
    chains_stats_dict: Dict[str, List[pd.DataFrame]],
    cols: List[str],
) -> pd.DataFrame:
    rows = []
    for mname, dfs in chains_stats_dict.items():
        row = {"Method": pretty_method_name(mname)}
        for col in cols:
            v = pooled_values(dfs, col)
            row[f"{col}_mean"] = float(np.mean(v))
            row[f"{col}_sd"] = float(np.std(v, ddof=1))
            row[f"{col}_q05"] = float(np.quantile(v, 0.05))
            row[f"{col}_q50"] = float(np.quantile(v, 0.50))
            row[f"{col}_q95"] = float(np.quantile(v, 0.95))
        rows.append(row)
    return pd.DataFrame(rows)


def save_method_hist_overlay(
    chains_stats_dict: Dict[str, List[pd.DataFrame]],
    outdir: str | Path,
    col: str,
    bins: int = 60,
    figsize: tuple[float, float] = (7.2, 5.2),
) -> None:
    plots_dir = ensure_plots_dir(outdir)
    fig, ax = plt.subplots(figsize=figsize)
    for mname, dfs in chains_stats_dict.items():
        ax.hist(
            pooled_values(dfs, col),
            bins=bins,
            density=True,
            alpha=0.35,
            label=pretty_method_name(mname),
        )
    ax.set_xlabel(pretty_observable_label(col))
    ax.set_ylabel("Density")
    ax.set_title(f"Cross-method pooled histogram overlay: {col}")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save_figure(fig, plots_dir / f"overlay_methods_{col}")


def save_method_ecdf_overlay(
    chains_stats_dict: Dict[str, List[pd.DataFrame]],
    outdir: str | Path,
    col: str,
    figsize: tuple[float, float] = (7.2, 5.2),
) -> None:
    plots_dir = ensure_plots_dir(outdir)
    fig, ax = plt.subplots(figsize=figsize)
    for mname, dfs in chains_stats_dict.items():
        vals = np.sort(pooled_values(dfs, col))
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals, y, label=pretty_method_name(mname))
    ax.set_xlabel(pretty_observable_label(col))
    ax.set_ylabel("ECDF")
    ax.set_title(f"Cross-method ECDF overlay: {col}")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save_figure(fig, plots_dir / f"overlay_methods_ecdf_{col}")


def save_chain_trace(
    df: pd.DataFrame,
    outdir: str | Path,
    method: str,
    col: str,
    chain_id: int = 0,
    figsize: tuple[float, float] = (7.2, 4.8),
) -> None:
    plots_dir = ensure_plots_dir(outdir)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(df[col].to_numpy())
    ax.set_xlabel("Iteration (post-burn, thinned)")
    ax.set_ylabel(pretty_observable_label(col))
    ax.set_title(f"{pretty_method_name(method)}: trace of {col} (chain {chain_id})")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save_figure(fig, plots_dir / f"trace_{method}_chain{chain_id}_{col}")


def save_two_panel_ecdf(
    chains_stats_dict: Dict[str, List[pd.DataFrame]],
    outdir: str | Path,
    col_left: str,
    col_right: str,
    filename: str,
    title_left: str | None = None,
    title_right: str | None = None,
    figsize: tuple[float, float] = (13.5, 5.2),
) -> None:
    plots_dir = ensure_plots_dir(outdir)
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for mname, dfs in chains_stats_dict.items():
        vals = np.sort(pooled_values(dfs, col_left))
        y = np.arange(1, len(vals) + 1) / len(vals)
        axes[0].plot(vals, y, label=pretty_method_name(mname))
    axes[0].set_xlabel(pretty_observable_label(col_left))
    axes[0].set_ylabel("ECDF")
    axes[0].set_title(title_left or f"(a) {pretty_observable_label(col_left)}")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    for mname, dfs in chains_stats_dict.items():
        vals = np.sort(pooled_values(dfs, col_right))
        y = np.arange(1, len(vals) + 1) / len(vals)
        axes[1].plot(vals, y, label=pretty_method_name(mname))
    axes[1].set_xlabel(pretty_observable_label(col_right))
    axes[1].set_ylabel("ECDF")
    axes[1].set_title(title_right or f"(b) {pretty_observable_label(col_right)}")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    save_figure(fig, plots_dir / filename)

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def pooled_values(chains_stats_list: list[pd.DataFrame], col: str) -> np.ndarray:
    return np.concatenate([df[col].values for df in chains_stats_list], axis=0)

def save_method_hist_overlay(chains_stats_dict: dict, outdir: str, col: str, bins: int = 60):
    """
    Overlay pooled histograms from multiple methods.
    """
    plt.figure()
    for mname, dfs in chains_stats_dict.items():
        vals = pooled_values(dfs, col)
        plt.hist(vals, bins=bins, density=True, alpha=0.35, label=mname)
    plt.xlabel(col)
    plt.ylabel("density")
    plt.title(f"Cross-method pooled histogram overlay: {col}")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)
    plt.savefig(os.path.join(outdir, "plots", f"overlay_methods_{col}.png"), dpi=200)
    plt.close()

def save_method_ecdf_overlay(chains_stats_dict: dict, outdir: str, col: str):
    """
    ECDF overlay for pooled samples (often clearer than histograms).
    """
    plt.figure()
    for mname, dfs in chains_stats_dict.items():
        vals = np.sort(pooled_values(dfs, col))
        y = np.linspace(0.0, 1.0, len(vals), endpoint=False)
        plt.plot(vals, y, label=mname)
    plt.xlabel(col)
    plt.ylabel("ECDF")
    plt.title(f"Cross-method ECDF overlay: {col}")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)
    plt.savefig(os.path.join(outdir, "plots", f"overlay_methods_ecdf_{col}.png"), dpi=200)
    plt.close()

def save_chain_trace(df: pd.DataFrame, outdir: str, method: str, col: str):
    """
    Trace plot for one chain (e.g., chain 0).
    """
    plt.figure()
    plt.plot(df[col].values)
    plt.xlabel("Iteration (post-burn, thinned)")
    plt.ylabel(col)
    plt.title(f"{method}: trace of {col} (chain 0)")
    plt.tight_layout()
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)
    plt.savefig(os.path.join(outdir, "plots", f"trace_{method}_{col}.png"), dpi=200)
    plt.close()

def save_within_method_hist_overlay(dfs: list[pd.DataFrame], outdir: str, method: str, col: str, bins: int = 40):
    """
    Overlay histograms across chains within one method.
    """
    plt.figure()
    for i, df in enumerate(dfs):
        plt.hist(df[col].values, bins=bins, density=True, alpha=0.35, label=f"chain {i}")
    plt.xlabel(col)
    plt.ylabel("density")
    plt.title(f"{method}: histogram overlay of {col}")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)
    plt.savefig(os.path.join(outdir, "plots", f"hist_{method}_{col}_overlay.png"), dpi=200)
    plt.close()

def compute_pooled_summary(chains_stats_dict: dict, cols: list[str]) -> pd.DataFrame:
    """
    mean/sd/quantiles for pooled post-burn samples (pooled over chains) per method.
    """
    rows = []
    for mname, dfs in chains_stats_dict.items():
        row = {"Method": mname}
        for col in cols:
            v = pooled_values(dfs, col)
            row[f"{col}_mean"] = float(np.mean(v))
            row[f"{col}_sd"] = float(np.std(v, ddof=1))
            row[f"{col}_q05"] = float(np.quantile(v, 0.05))
            row[f"{col}_q50"] = float(np.quantile(v, 0.50))
            row[f"{col}_q95"] = float(np.quantile(v, 0.95))
        rows.append(row)
    return pd.DataFrame(rows)
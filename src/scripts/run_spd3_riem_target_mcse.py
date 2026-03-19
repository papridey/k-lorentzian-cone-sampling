
"""
Run SPD(3) affine-invariant Riemannian target experiment:
  - geom_MALA (Riemannian drift in S-coordinates)
  - naive_Euclid_drift_in_S (Euclidean drift mapped into S-coordinates)

Target (Riemannian base measure):
  pi(dX) ∝ exp(-Phi(X)) vol_g(dX) on SPD(3),
  Phi(X) = (lambda/2) d_g(X,X0)^2 - beta logdet(X) + (kappa/2)(tr(X)-1)^2

Outputs:
  <outdir>/results/summary.csv
  <outdir>/results/pooled_method_summary_stats.csv
  <outdir>/results/rho_proxy.csv
  <outdir>/results/mcse_zscore_comparison.csv (+ .tex)
  <outdir>/plots/overlay_methods_*.png and overlay_methods_ecdf_*.png
"""

from __future__ import annotations

import os
import time
import argparse
import numpy as np
import pandas as pd

# If running from repo root, this makes `import src.*` work reliably.
# (Alternative is to install package in editable mode: pip install -e .)
import sys
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.spd_ai_geometry import spd_project  # noqa: E402
from src.targets import TargetSPD_AI        # noqa: E402
from src.samplers import run_chain_geom_MALA, run_chain_naive_Euclid_drift_in_S  # noqa: E402
from src.diagnostics import (
    split_rhat,
    ess_1d,
    compute_rho_suite,
    build_mcse_z_table,
)  # noqa: E402
from src.plotting import (
    save_method_overlay,
    save_method_ecdf_overlay,
    compute_pooled_summary,
)  # noqa: E402


def _make_outdir(prefix: str) -> str:
    outdir = f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "results"), exist_ok=True)
    return outdir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, default="spd3_riem_target_mcse",
                        help="Output directory prefix")
    parser.add_argument("--n_chains", type=int, default=4)
    parser.add_argument("--N", type=int, default=15000)
    parser.add_argument("--burn", type=int, default=3000)
    parser.add_argument("--thin", type=int, default=5)
    parser.add_argument("--h_geom", type=float, default=8e-3)
    parser.add_argument("--h_naive", type=float, default=8e-3)
    parser.add_argument("--lambda_", type=float, default=10.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--kappa", type=float, default=50.0)
    parser.add_argument("--seed0", type=int, default=0, help="First chain seed (uses seed0..seed0+n_chains-1)")
    parser.add_argument("--rho_linear_tests", type=int, default=5)
    args = parser.parse_args()

    outdir = _make_outdir(args.prefix)

    # Target parameters
    X0 = np.eye(3)
    target = TargetSPD_AI(lambda_=args.lambda_, beta=args.beta, kappa=args.kappa, X0=X0)

    # Methods
    methods = {
        "geom_MALA": (run_chain_geom_MALA, args.h_geom),
        "naive_Euclid_drift_in_S": (run_chain_naive_Euclid_drift_in_S, args.h_naive),
    }

    seeds = list(range(args.seed0, args.seed0 + args.n_chains))

    # Store results
    chains_stats: dict[str, list[pd.DataFrame]] = {m: [] for m in methods}
    chains_meta: dict[str, list[dict]] = {m: [] for m in methods}
    chains_samples: dict[str, list[np.ndarray]] = {m: [] for m in methods}  # matrices (post-burn, thinned)

    # Run chains ONCE (store matrices for rho proxy from the same run)
    for mname, (runner, h_use) in methods.items():
        for c in range(args.n_chains):
            rng = np.random.default_rng(seeds[c])
            X_init = spd_project(np.eye(3) + 0.20 * rng.normal(size=(3, 3)))
            mats, df, meta = runner(
                rng,
                target,
                N=args.N,
                burn=args.burn,
                thin=args.thin,
                h=h_use,
                X_init=X_init,
                store_mats=True,
            )
            chains_stats[mname].append(df)
            chains_meta[mname].append(meta)
            chains_samples[mname].append(mats)

    # -------------------------
    # Summary diagnostics table
    # -------------------------
    obs = ["logdet", "lmin", "d2", "tr"]
    summary_rows = []
    for mname in methods:
        metas = chains_meta[mname]
        dfs = chains_stats[mname]
        acc = np.array([m["acc_rate"] for m in metas], dtype=float)
        rt = np.array([m["elapsed"] for m in metas], dtype=float)

        row = {
            "Method": mname,
            "Runtime_sec_per_chain": float(rt.mean()),
            "Acc_mean": float(acc.mean()),
            "Acc_sd": float(acc.std(ddof=1)),
        }
        for nm in obs:
            series = [df[nm].values for df in dfs]
            row[f"Rhat_{nm}"] = split_rhat(series)
            ess_list = [ess_1d(s) / m["elapsed"] for s, m in zip(series, metas)]
            row[f"ESSsec_{nm}"] = float(np.mean(ess_list))
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(outdir, "results", "summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    # -------------------------
    # Pooled marginal summaries
    # -------------------------
    pooled_df = compute_pooled_summary(chains_stats, obs)
    pooled_csv = os.path.join(outdir, "results", "pooled_method_summary_stats.csv")
    pooled_df.to_csv(pooled_csv, index=False)

    # -------------------------
    # rho proxy (same samples)
    # -------------------------
    rho_rows = []
    for mname in methods:
        pooled_samples = np.concatenate(chains_samples[mname], axis=0)
        rho = compute_rho_suite(
            pooled_samples,
            target,
            n_linear_tests=args.rho_linear_tests,
            seed=123,
        )
        rho["Method"] = mname
        rho_rows.append(rho)
    rho_df = pd.DataFrame(rho_rows)
    rho_csv = os.path.join(outdir, "results", "rho_proxy.csv")
    rho_df.to_csv(rho_csv, index=False)

    # -------------------------
    # MCSE + z-score table
    # -------------------------
    mcse_cols = ["tr", "logdet", "d2", "lmin"]
    mcse_df = build_mcse_z_table(chains_stats, "geom_MALA", "naive_Euclid_drift_in_S", mcse_cols)
    mcse_csv = os.path.join(outdir, "results", "mcse_zscore_comparison.csv")
    mcse_df.to_csv(mcse_csv, index=False)
    mcse_tex = os.path.join(outdir, "results", "mcse_zscore_comparison.tex")
    with open(mcse_tex, "w") as f:
        f.write(mcse_df.to_latex(index=False, float_format=lambda x: f"{x:.6g}"))

    # -------------------------
    # Cross-method overlays
    # -------------------------
    for nm in obs:
        save_method_overlay(chains_stats, outdir, nm, bins=60)
        save_method_ecdf_overlay(chains_stats, outdir, nm)

    # -------------------------
    # Print summary
    # -------------------------
    print("============================================================")
    print("Output dir:", outdir)
    print("Saved summary CSV:", summary_csv)
    print("Saved pooled stats CSV:", pooled_csv)
    print("Saved rho CSV:", rho_csv)
    print("Saved MCSE/z-score CSV:", mcse_csv)
    print("Saved plots in:", os.path.join(outdir, "plots"))
    print("============================================================")
    print(summary_df)
    print("============================================================")
    print("Pooled marginal summaries:")
    print(pooled_df)
    print("============================================================")
    print("rho proxy:")
    print(rho_df)
    print("============================================================")
    print("MCSE/z-score comparison:")
    print(mcse_df)
    print("============================================================")


if __name__ == "__main__":
    main()
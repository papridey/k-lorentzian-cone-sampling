from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Method display names
# ------------------------------------------------------------
METHOD_ORDER = [
    "cone_geom_mala",
    "Euclidean_MALA",
    "generic_RMALA",
]

METHOD_LABELS = {
    "cone_geom_mala": r"\textsc{ConeMALA}",
    "Euclidean_MALA": "Euclidean MALA",
    "generic_RMALA": "Generic RMALA",
}


def find_summary_files(outdir: Path, d: int, m_list: list[int]) -> list[Path]:
    """
    Search for reviewer-friendly summary files in two locations:

    1. outdir/summary_d5_m20.csv
    2. outdir/graph_posterior_d5_m20/results/summary_d5_m20.csv
    3. fallback: outdir/graph_posterior_d5_m20/results/summary_multichain.csv
    """
    files = []

    for m in m_list:
        candidates = [
            outdir / f"summary_d{d}_m{m}.csv",
            outdir / f"graph_posterior_d{d}_m{m}" / "results" / f"summary_d{d}_m{m}.csv",
            outdir / f"graph_posterior_d{d}_m{m}" / "results" / "summary_multichain.csv",
        ]

        found = None
        for p in candidates:
            if p.exists():
                found = p
                break

        if found is None:
            print(f"[warning] No summary found for d={d}, m={m}. Skipping this m.")
        else:
            print(f"[found] d={d}, m={m}: {found}")
            files.append(found)

    return files


def load_scaling_summary(outdir: Path, d: int, m_list: list[int]) -> pd.DataFrame:
    files = find_summary_files(outdir, d, m_list)

    if not files:
        raise FileNotFoundError(
            f"No summary files found in {outdir}. "
            "Run the experiments first or check the output directory."
        )

    dfs = []
    for f in files:
        df = pd.read_csv(f)

        # Some older summaries may not include d,m columns.
        if "d" not in df.columns or "m" not in df.columns:
            # Try to parse from filename or parent folder.
            name = str(f)
            parsed_m = None
            for m in m_list:
                if f"m{m}" in name or f"m-{m}" in name:
                    parsed_m = m
                    break
            df["d"] = d
            df["m"] = parsed_m

        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    # Keep only requested methods and sort.
    combined = combined[combined["Method"].isin(METHOD_ORDER)].copy()
    combined["Method"] = pd.Categorical(
        combined["Method"],
        categories=METHOD_ORDER,
        ordered=True,
    )
    combined = combined.sort_values(["m", "Method"]).reset_index(drop=True)

    return combined


def compute_worst_rhat(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a worst_Rhat column over the main observables.
    We include rel_W_error, test_nll, logdet_Q, lambda_min_Q, and Phi
    if present.
    """
    rhat_cols = [
        "rel_W_error_Rhat",
        "test_nll_Rhat",
        "logdet_Q_Rhat",
        "lambda_min_Q_Rhat",
        "Phi_Rhat",
    ]
    available = [c for c in rhat_cols if c in df.columns]

    if not available:
        raise ValueError("No Rhat columns found in the summary CSV files.")

    df = df.copy()
    df["worst_Rhat"] = df[available].max(axis=1)
    return df


def plot_esssec_vs_m(df: pd.DataFrame, outdir: Path, ess_col: str) -> None:
    """
    Plot Rel. W ESS/sec versus m.
    """
    fig, ax = plt.subplots(figsize=(6.8, 4.6))

    for method in METHOD_ORDER:
        sub = df[df["Method"] == method].sort_values("m")
        if sub.empty:
            continue

        ax.plot(
            sub["m"],
            sub[ess_col],
            marker="o",
            linewidth=2,
            markersize=6,
            label=METHOD_LABELS.get(method, method),
        )

    ax.set_xlabel(r"Graph size \(m\)")
    ax.set_ylabel(r"Relative \(W\) ESS/sec")
    ax.set_title(r"(a) Sampling efficiency versus graph size")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_xticks(sorted(df["m"].dropna().unique()))

    fig.tight_layout()

    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "scaling_relW_ESSsec_vs_m.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "scaling_relW_ESSsec_vs_m.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_rhat_vs_m(df: pd.DataFrame, outdir: Path, use_phi_only: bool = False) -> None:
    """
    Plot either Phi-Rhat or worst split-Rhat versus m.
    """
    fig, ax = plt.subplots(figsize=(6.8, 4.6))

    if use_phi_only:
        y_col = "Phi_Rhat"
        y_label = r"\(\widehat R_{\Phi}\)"
        title = r"(b) Energy split-\(\widehat R\) versus graph size"
        fname = "scaling_phi_Rhat_vs_m"
    else:
        y_col = "worst_Rhat"
        y_label = r"Worst split-\(\widehat R\)"
        title = r"(b) Worst split-\(\widehat R\) versus graph size"
        fname = "scaling_worst_Rhat_vs_m"

    for method in METHOD_ORDER:
        sub = df[df["Method"] == method].sort_values("m")
        if sub.empty:
            continue

        ax.plot(
            sub["m"],
            sub[y_col],
            marker="o",
            linewidth=2,
            markersize=6,
            label=METHOD_LABELS.get(method, method),
        )

    # Reference lines
    ax.axhline(1.05, linestyle="--", linewidth=1)
    ax.axhline(1.10, linestyle=":", linewidth=1)

    ax.set_xlabel(r"Graph size \(m\)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_xticks(sorted(df["m"].dropna().unique()))

    fig.tight_layout()

    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{fname}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / f"{fname}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_two_panel_scaling(
    df: pd.DataFrame,
    outdir: Path,
    ess_col: str = "rel_W_error_ESSsec",
    use_phi_only: bool = False,
) -> None:
    """
    Make a two-panel figure:
      (a) Rel. W ESS/sec versus m
      (b) worst split-Rhat or Phi-Rhat versus m
    """
    if use_phi_only:
        rhat_col = "Phi_Rhat"
        rhat_label = r"\(\widehat R_{\Phi}\)"
        fname = "scaling_two_panel_relW_ESSsec_phiRhat"
        title_right = r"(b) Energy split-\(\widehat R\)"
    else:
        rhat_col = "worst_Rhat"
        rhat_label = r"Worst split-\(\widehat R\)"
        fname = "scaling_two_panel_relW_ESSsec_worstRhat"
        title_right = r"(b) Worst split-\(\widehat R\)"

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    for method in METHOD_ORDER:
        sub = df[df["Method"] == method].sort_values("m")
        if sub.empty:
            continue

        label = METHOD_LABELS.get(method, method)

        axes[0].plot(
            sub["m"],
            sub[ess_col],
            marker="o",
            linewidth=2,
            markersize=6,
            label=label,
        )

        axes[1].plot(
            sub["m"],
            sub[rhat_col],
            marker="o",
            linewidth=2,
            markersize=6,
            label=label,
        )

    axes[0].set_xlabel(r"Graph size \(m\)")
    axes[0].set_ylabel(r"Relative \(W\) ESS/sec")
    axes[0].set_title(r"(a) Sampling efficiency")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(sorted(df["m"].dropna().unique()))

    axes[1].set_xlabel(r"Graph size \(m\)")
    axes[1].set_ylabel(rhat_label)
    axes[1].set_title(title_right)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(sorted(df["m"].dropna().unique()))
    axes[1].axhline(1.05, linestyle="--", linewidth=1)
    axes[1].axhline(1.10, linestyle=":", linewidth=1)

    axes[0].legend(fontsize=9)
    axes[1].legend(fontsize=9)

    fig.tight_layout()

    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{fname}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / f"{fname}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate d=5 scaling plots from saved summary_d5_m*.csv files."
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="cone_mala_d5_m_sweep_outputs",
        help="Directory containing summary files.",
    )
    parser.add_argument("--d", type=int, default=5)
    parser.add_argument(
        "--m_list",
        type=str,
        default="20,50,100",
        help="Comma-separated m list, e.g. 20,50,100.",
    )
    parser.add_argument(
        "--plotdir",
        type=str,
        default="",
        help="Optional output directory for plots. Defaults to outdir/scaling_plots.",
    )
    parser.add_argument(
        "--use_phi_only",
        action="store_true",
        help="Use Phi_Rhat instead of worst_Rhat in panel (b).",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    plotdir = Path(args.plotdir) if args.plotdir else outdir / "scaling_plots"
    m_list = [int(x.strip()) for x in args.m_list.split(",") if x.strip()]

    df = load_scaling_summary(outdir, args.d, m_list)
    df = compute_worst_rhat(df)

    required = ["rel_W_error_ESSsec", "Phi_Rhat"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Save the combined plotting data.
    plotdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(plotdir / f"scaling_plot_data_d{args.d}.csv", index=False)

    # Print compact view.
    cols = [
        "d",
        "m",
        "Method",
        "accept_mean",
        "rel_W_error_ESSsec",
        "test_nll_ESSsec",
        "Phi_Rhat",
        "worst_Rhat",
    ]
    cols = [c for c in cols if c in df.columns]
    print("\nScaling plot data:")
    print(df[cols].to_string(index=False))

    # Individual plots.
    plot_esssec_vs_m(df, plotdir, ess_col="rel_W_error_ESSsec")
    plot_rhat_vs_m(df, plotdir, use_phi_only=args.use_phi_only)

    # Two-panel plot.
    plot_two_panel_scaling(
        df,
        plotdir,
        ess_col="rel_W_error_ESSsec",
        use_phi_only=args.use_phi_only,
    )

    print(f"\nSaved plots to: {plotdir}")


if __name__ == "__main__":
    main()
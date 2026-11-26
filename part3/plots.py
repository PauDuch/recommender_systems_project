import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Part 3 Track A Summary Plots (Frontier, Disaggregation & Coverage Diagnosis)


def plot_track_a_frontier(track_a_csv: Path, output_dir: Path) -> Path:
    """Generates wide 2-panel figure: Accuracy-Diversity Frontier and Calibration/Coverage vs Lambda."""
    df = pd.read_csv(track_a_csv).sort_values("Lambda", ascending=False).reset_index(drop=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))

    # Left panel: Accuracy (NDCG@10) vs. Intra-List Diversity (ILD) Pareto Frontier
    ax1.plot(
        df["Intra-list diversity"],
        df["NDCG@10"],
        marker="o",
        linewidth=2.0,
        markersize=7,
        color="#2ca02c",
        zorder=3,
    )
    for _, row in df.iterrows():
        lam = round(float(row["Lambda"]), 3)
        if lam not in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3, 0.0):
            continue
        # Offset labels cleanly so none overlap
        if lam == 1.0:
            xytext = (-8, -13)
        elif lam == 0.9:
            xytext = (-6, 6)
        elif lam == 0.8:
            xytext = (0, 7)
        elif lam == 0.7:
            xytext = (4, 5)
        elif lam == 0.0:
            xytext = (-28, -11)
        else:
            xytext = (5, 4)
        ax1.annotate(
            f"λ={lam:.1f}",
            (row["Intra-list diversity"], row["NDCG@10"]),
            xytext=xytext,
            textcoords="offset points",
            fontsize=8,
            fontweight="bold" if lam in (1.0, 0.8, 0.7, 0.6, 0.5) else "normal",
        )

    ax1.set_xlabel("Intra-List Diversity (Cosine Genre Distance)")
    ax1.set_ylabel("Test NDCG@10")
    ax1.set_title("Accuracy–Diversity Frontier (MMR on Mult-VAE)")
    ax1.set_ylim(0.0565, 0.0725)
    ax1.set_xlim(0.59, 0.915)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Right panel: Calibration Error & Catalogue Coverage across Lambda
    color_cal = "#1f77b4"
    color_cov = "#d62728"

    ax2.plot(
        df["Lambda"],
        df["Calibration error"],
        marker="s",
        linewidth=1.8,
        color=color_cal,
        label="Calibration Error (KL ↓)",
    )
    ax2.set_xlabel("MMR Trade-off Parameter λ (1.0 = Base Mult-VAE → 0.0 = Pure Diversity)")
    ax2.set_ylabel("Calibration Error (KL Divergence)", color=color_cal)
    ax2.tick_params(axis="y", labelcolor=color_cal)
    ax2.invert_xaxis()
    ax2.grid(True, linestyle="--", alpha=0.4)

    ax2_twin = ax2.twinx()
    ax2_twin.plot(
        df["Lambda"],
        df["Catalogue coverage"],
        marker="^",
        linewidth=1.8,
        linestyle="--",
        color=color_cov,
        label="Catalogue Coverage (%)",
    )
    ax2_twin.set_ylabel("Catalogue Coverage (%)", color=color_cov)
    ax2_twin.tick_params(axis="y", labelcolor=color_cov)

    # Combine legends from both y-axes
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper center", fontsize=8, frameon=True)
    ax2.set_title("Calibration Error & Coverage across λ")

    plt.tight_layout()
    out_path = output_dir / "part3_track_a_frontier.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_track_a_analysis(disagg_csv: Path, diag_csv: Path, output_dir: Path) -> Path:
    """Generates wide 2-panel figure: User-Activity Calibration Error & Simple Coverage Bar Comparison."""
    disagg_df = pd.read_csv(disagg_csv)
    diag_df = pd.read_csv(diag_csv)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))

    # Left panel: Calibration Error across User Activity Tertiles (with rating thresholds on x-axis)
    groups = ["Low Activity", "Medium Activity", "High Activity"]
    group_labels = [
        "Low Activity\n(≤ 45 ratings)",
        "Medium Activity\n(46–124 ratings)",
        "High Activity\n(> 124 ratings)",
    ]
    systems = [
        ("Item-kNN (k=100, 2nd-Best)", "Item-kNN (k=100)", "#7f7f7f"),
        ("Mult-VAE (Base, lambda=1.0)", "Mult-VAE (λ=1.0)", "#1f77b4"),
        ("Mult-VAE + MMR (lambda=0.8)", "MMR (λ=0.8)", "#2ca02c"),
        ("Mult-VAE + MMR (lambda=0.7)", "MMR (λ=0.7)", "#ff7f0e"),
    ]
    x = np.arange(len(groups))
    width = 0.20

    for idx, (sys_key, label, color) in enumerate(systems):
        sub = disagg_df[disagg_df["System"] == sys_key].set_index("Group")
        vals = [float(sub.loc[g, "Calibration error"]) for g in groups]
        offset = (idx - 1.5) * width
        bars = ax1.bar(x + offset, vals, width=width, label=label, color=color, alpha=0.9)
        for bar in bars:
            h = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 0.012,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.0,
            )

    ax1.set_xticks(x)
    ax1.set_xticklabels(group_labels)
    ax1.set_ylabel("Calibration Error (KL Divergence ↓)")
    ax1.set_title("Genre Calibration Error by User Activity Group")
    ax1.set_ylim(0.0, 1.15)
    ax1.legend(loc="upper left", fontsize=7.5, ncol=2, frameon=True)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.4)

    # Right panel: Simple grouped bar chart comparing Catalogue Coverage (%) vs Top-5 Blockbuster Share (%)
    selected_configs = [
        ("Standard Genre MMR (lambda=1.0)", "Base Mult-VAE\n(λ=1.0)"),
        ("Standard Genre MMR (lambda=0.7)", "Standard MMR\n(λ=0.7)"),
        ("Standard Genre MMR (lambda=0.5)", "Standard MMR\n(λ=0.5)"),
        ("Popularity-TieBreak MMR (lambda=0.7, gamma=0.25)", "Pop-TieBreak\n(λ=0.7, γ=0.25)"),
        ("Popularity-TieBreak MMR (lambda=0.7, gamma=0.35)", "Pop-TieBreak\n(λ=0.7, γ=0.35)"),
    ]
    diag_indexed = diag_df.set_index("Configuration")
    x2 = np.arange(len(selected_configs))
    w2 = 0.34

    top5_vals = [float(diag_indexed.loc[k, "Within-genre top-5 share (%)"]) for k, _ in selected_configs]
    cov_vals = [float(diag_indexed.loc[k, "Catalogue coverage"]) for k, _ in selected_configs]
    x2_labels = [lbl for _, lbl in selected_configs]

    bars_top5 = ax2.bar(
        x2 - w2 / 2,
        top5_vals,
        width=w2,
        color="#d62728",
        alpha=0.88,
        label="Top-5 Blockbusters per Genre (% ↓)",
    )
    bars_cov = ax2.bar(
        x2 + w2 / 2,
        cov_vals,
        width=w2,
        color="#1f77b4",
        alpha=0.92,
        label="Catalogue Coverage (% ↑)",
    )

    for bar in bars_top5:
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.5,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            fontsize=6.8,
            color="#a51d1d",
            fontweight="bold",
        )
    for bar in bars_cov:
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.5,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            fontsize=6.8,
            color="#155785",
            fontweight="bold",
        )

    # Vertical divider separating Standard MMR (problem) from Pop-TieBreak MMR (fix)
    ax2.axvline(2.5, color="#555555", linestyle=":", linewidth=1.3)

    ax2.set_xticks(x2)
    ax2.set_xticklabels(x2_labels, fontsize=7.8)
    ax2.set_ylabel("Percentage (%)")
    ax2.set_title("Why Standard MMR Drops Coverage & How Tie-Breaking Fixes It")
    ax2.set_ylim(25.0, 58.0)
    ax2.legend(loc="upper left", fontsize=7.5, frameon=True)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    out_path = output_dir / "part3_track_a_analysis.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_part3_plots(output_dir: Path) -> list[Path]:
    """Generates the Track A summary figures from saved CSV outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    track_a_csv = output_dir / "track_a_results.csv"
    if track_a_csv.exists():
        generated.append(plot_track_a_frontier(track_a_csv, output_dir))

    disagg_csv = output_dir / "track_a_comparison_and_disagg.csv"
    diag_csv = output_dir / "track_a_coverage_diagnosis.csv"
    if disagg_csv.exists() and diag_csv.exists():
        generated.append(plot_track_a_analysis(disagg_csv, diag_csv, output_dir))

    for p in generated:
        print(f"Saved Part 3 plot: {p}")
    return generated


def main():
    parser = argparse.ArgumentParser(description="Generate Part 3 Track A Plots from Saved CSVs")
    parser.add_argument("--output-dir", type=str, default="part3/outputs", help="Directory containing Part 3 CSVs")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent
    out_path = Path(args.output_dir) if Path(args.output_dir).is_absolute() else root_dir / args.output_dir
    generate_part3_plots(out_path)


if __name__ == "__main__":
    main()

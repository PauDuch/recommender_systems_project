import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# Part 3 Track A Summary Plots (Accuracy-Diversity Frontier & Beyond-Accuracy Trade-offs)


def plot_track_a_frontier(track_a_csv: Path, output_dir: Path) -> Path:
    """Generates wide 2-panel figure: Accuracy-Diversity Frontier and Calibration/Coverage/Gini vs Lambda."""
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


def generate_part3_plots(output_dir: Path) -> list[Path]:
    """Generates the Track A summary figure from track_a_results.csv."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    track_a_csv = output_dir / "track_a_results.csv"
    if track_a_csv.exists():
        generated.append(plot_track_a_frontier(track_a_csv, output_dir))

    for p in generated:
        print(f"Saved Part 3 plot: {p}")
    return generated


def main():
    parser = argparse.ArgumentParser(description="Generate Part 3 Track A Plots from Saved CSV")
    parser.add_argument("--output-dir", type=str, default="part3/outputs", help="Directory containing Part 3 CSVs")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent
    out_path = Path(args.output_dir) if Path(args.output_dir).is_absolute() else root_dir / args.output_dir
    generate_part3_plots(out_path)


if __name__ == "__main__":
    main()

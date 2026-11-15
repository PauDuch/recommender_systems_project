# Part 2 Summary Plots (Wide, Report-Ready Figures from Saved CSVs)

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_accuracy_vs_coverage(test_csv: Path, output_dir: Path) -> Path:
    """Generates wide 2-panel figure: NDCG@10 (with 95% CI) and NDCG@10 vs. Coverage."""
    df = pd.read_csv(test_csv).sort_values("NDCG@10 Mean", ascending=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))

    # Left panel: NDCG@10 horizontal bars with 95% CI error bars
    colors = ["#4C72B0" if "Mult-VAE" not in m else "#2ca02c" for m in df["Model"]]
    ax1.barh(
        df["Model"],
        df["NDCG@10 Mean"],
        xerr=df["NDCG@10 CI95"],
        color=colors,
        alpha=0.9,
        capsize=3,
        edgecolor="black",
        linewidth=0.5,
    )
    ax1.set_xlabel("Test NDCG@10 (Mean ± 95% CI)")
    ax1.set_title("Test Ranking Accuracy")
    ax1.grid(axis="x", linestyle="--", alpha=0.4)
    for idx, (_, row) in enumerate(df.iterrows()):
        ax1.text(row["NDCG@10 Mean"] + row["NDCG@10 CI95"] + 0.001, idx, f"{row['NDCG@10 Mean']:.4f}", va="center", fontsize=8)

    # Right panel: Accuracy vs. Catalogue Coverage scatter
    ax2.scatter(
        df["Coverage (%) Mean"],
        df["NDCG@10 Mean"],
        s=90,
        color="#dd8452",
        edgecolor="black",
        zorder=3,
    )
    for _, row in df.iterrows():
        x_off = -3.0 if row["Coverage (%) Mean"] > 80 else 1.5
        ha = "right" if row["Coverage (%) Mean"] > 80 else "left"
        ax2.annotate(
            row["Model"],
            (row["Coverage (%) Mean"], row["NDCG@10 Mean"]),
            xytext=(x_off, 4),
            textcoords="offset points",
            fontsize=8,
            ha=ha,
        )
    ax2.set_xlabel("Catalogue Coverage (%)")
    ax2.set_ylabel("Test NDCG@10")
    ax2.set_title("Accuracy vs. Catalogue Coverage Trade-off")
    ax2.set_xlim(-2, 98)
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    out_path = output_dir / "part2_accuracy_vs_coverage.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_disagg_and_sparsity(disagg_csv: Path, sparsity_csv: Path, output_dir: Path) -> Path:
    """Generates wide 2-panel figure: Item Popularity Disaggregation and Sparsity Ablation."""
    dis_df = pd.read_csv(disagg_csv)
    spa_df = pd.read_csv(sparsity_csv)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))

    # Left panel: Item Popularity Disaggregation (NDCG@10)
    item_df = dis_df[dis_df["Breakdown dimension"] == "Item Popularity"]
    groups = ["Popular (Top 20%)", "Mid-Tier (20-50%)", "Long-Tail (Bottom 50%)"]
    short_labels = ["Popular\n(Top 20%)", "Mid-Tier\n(20-50%)", "Long-Tail\n(Bottom 50%)"]
    models = ["Mult-VAE", "Item-kNN (k=100)", "BPR-MF", "Most Popular (Pos)", "MF-SGD", "Content-Based"]
    x = np.arange(len(groups))
    width = 0.13

    for i, model in enumerate(models):
        vals = [
            item_df[(item_df["Model"] == model) & (item_df["Group"] == g)]["NDCG@10"].values[0]
            for g in groups
        ]
        ax1.bar(x + (i - 2.5) * width, vals, width, label=model, edgecolor="black", linewidth=0.3)

    ax1.set_xticks(x)
    ax1.set_xticklabels(short_labels, fontsize=8)
    ax1.set_ylabel("NDCG@10")
    ax1.set_title("Disaggregated NDCG@10 by Item Popularity")
    ax1.legend(fontsize=7, frameon=True, loc="upper right")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    # Right panel: Sparsity Ablation Curve
    spa_df["PctNum"] = spa_df["Training interactions (%)"].str.rstrip("%").astype(int)
    spa_df = spa_df.sort_values("PctNum", ascending=False)
    for model in models:
        sub = spa_df[spa_df["Model"] == model]
        ax2.plot(sub["Training interactions (%)"], sub["NDCG@10"], marker="o", linewidth=1.8, label=model)

    ax2.set_xlabel("Retained Training Interactions")
    ax2.set_ylabel("Test NDCG@10")
    ax2.set_title("Sparsity Ablation (100% -> 50% -> 20%)")
    ax2.legend(fontsize=7, frameon=True, loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    out_path = output_dir / "part2_disagg_and_sparsity.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_part2_plots(output_dir: Path) -> list[Path]:
    """Generates the 2 summary figures for Part 2 from saved CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    test_csv = output_dir / "test_results_part2.csv"
    disagg_csv = output_dir / "disaggregation_part2.csv"
    sparsity_csv = output_dir / "sparsity_part2.csv"

    if test_csv.exists():
        generated.append(plot_accuracy_vs_coverage(test_csv, output_dir))
    if disagg_csv.exists() and sparsity_csv.exists():
        generated.append(plot_disagg_and_sparsity(disagg_csv, sparsity_csv, output_dir))

    for p in generated:
        print(f"Saved Part 2 plot: {p}")
    return generated


def main():
    parser = argparse.ArgumentParser(description="Generate Part 2 Report Plots from Saved CSVs")
    parser.add_argument("--output-dir", type=str, default="part2/outputs", help="Directory containing Part 2 CSVs")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent
    out_path = Path(args.output_dir) if Path(args.output_dir).is_absolute() else root_dir / args.output_dir
    generate_part2_plots(out_path)


if __name__ == "__main__":
    main()

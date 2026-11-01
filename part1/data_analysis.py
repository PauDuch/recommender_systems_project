from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_statistics(ratings: pd.DataFrame, movies: pd.DataFrame, users: pd.DataFrame, output_dir: Path | str,) -> dict:
    """Computes descriptive statistics"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Core counts
    n_users = ratings["user_id"].nunique()
    n_items_rated = ratings["item_id"].nunique()
    n_items_catalog = len(movies)
    n_interactions = len(ratings)

    # Densities
    density_rated = (n_interactions / (n_users * n_items_rated)) * 100
    density_catalog = (n_interactions / (n_users * n_items_catalog)) * 100

    # User activity distribution (interactions per user)
    user_counts = ratings.groupby("user_id").size()
    user_stats = {
        "min": int(user_counts.min()),
        "max": int(user_counts.max()),
        "mean": float(user_counts.mean()),
        "median": float(user_counts.median()),
        "std": float(user_counts.std()),
    }

    # Item activity distribution (interactions per item)
    item_counts = ratings.groupby("item_id").size()
    item_stats = {
        "min": int(item_counts.min()),
        "max": int(item_counts.max()),
        "mean": float(item_counts.mean()),
        "median": float(item_counts.median()),
        "std": float(item_counts.std()),
    }

    # Rating score distribution
    rating_stats = {
        "mean": float(ratings["rating"].mean()),
        "median": float(ratings["rating"].median()),
        "std": float(ratings["rating"].std()),
    }

    # Temporal range
    t_min = ratings["datetime"].min()
    t_max = ratings["datetime"].max()
    t_span_days = (t_max - t_min).days

    summary_text = (
        "=============================================================\n"
        "           MOVIELENS 1M - DATA CHARACTERIZATION SUMMARY      \n"
        "=============================================================\n\n"
        "1. CORE COUNTS & DENSITY\n"
        f"  - Total Ratings (|R|)          : {n_interactions:,}\n"
        f"  - Unique Users (|U|)           : {n_users:,}\n"
        f"  - Rated Items (|I_rated|)      : {n_items_rated:,}\n"
        f"  - Catalog Items (|I_catalog|)  : {n_items_catalog:,}\n"
        f"  - Matrix Density (rated items) : {density_rated:.2f}%\n"
        f"  - Matrix Density (catalog)     : {density_catalog:.2f}%\n\n"
        "2. USER ACTIVITY (Ratings per User)\n"
        f"  - Min / Max                    : {user_stats['min']} / {user_stats['max']:,}\n"
        f"  - Mean                         : {user_stats['mean']:.2f}\n"
        f"  - Median                       : {user_stats['median']:.1f}\n"
        f"  - Std                          : {user_stats['std']:.2f}\n\n"
        "3. ITEM POPULARITY (Ratings per Item)\n"
        f"  - Min / Max                    : {item_stats['min']} / {item_stats['max']:,}\n"
        f"  - Mean                         : {item_stats['mean']:.2f}\n"
        f"  - Median                       : {item_stats['median']:.1f}\n"
        f"  - Std                          : {item_stats['std']:.2f}\n\n"
        "4. RATING SCORES\n"
        f"  - Mean                         : {rating_stats['mean']:.2f}\n"
        f"  - Median                       : {rating_stats['median']:.1f}\n"
        f"  - Std                          : {rating_stats['std']:.2f}\n\n"
        "5. TEMPORAL SPAN\n"
        f"  - Earliest interaction         : {t_min.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
        f"  - Latest interaction           : {t_max.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
        f"  - Total span (days)            : {t_span_days:,} days (~{t_span_days / 365.25:.1f} years)\n"
        "=============================================================\n"
    )

    # Print summary to console
    print(summary_text)

    return {
        "n_users": n_users,
        "n_items_rated": n_items_rated,
        "n_items_catalog": n_items_catalog,
        "n_interactions": n_interactions,
        "density_rated": density_rated,
        "user_stats": user_stats,
        "item_stats": item_stats,
        "rating_stats": rating_stats,
        "time_span_days": t_span_days,
    }


def generate_plots(ratings: pd.DataFrame, output_dir: Path | str) -> None:
    """Generates the four plots for Part 1."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure style
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

    # 1. Item Long-Tail Plot (Rank vs Interaction count)
    item_counts = ratings.groupby("item_id").size().sort_values(ascending=False).values
    ranks = np.arange(1, len(item_counts) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    # Linear scale
    ax1.plot(ranks, item_counts, color="#1f77b4", linewidth=2)
    ax1.fill_between(ranks, item_counts, color="#1f77b4", alpha=0.3)
    ax1.set_title("Item Interaction Distribution (Linear Scale)")
    ax1.set_xlabel("Item Popularity Rank")
    ax1.set_ylabel("Number of Ratings")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Log-Log scale
    ax2.loglog(ranks, item_counts, color="#d62728", linewidth=2)
    ax2.set_title("Item Long Tail (Log-Log Scale)")
    ax2.set_xlabel("Item Popularity Rank (log)")
    ax2.set_ylabel("Number of Ratings (log)")
    ax2.grid(True, which="both", linestyle="--", alpha=0.6)

    fig.suptitle("Item Long-Tail Distribution (Lecture 5.1, slide 5)", fontsize=13, fontweight="bold")
    plot_path1 = output_dir / "item_long_tail.png"
    plt.savefig(plot_path1, dpi=300)
    plt.close()
    print(f"Saved: {plot_path1.name}")

    # 2. User Activity Distribution
    user_counts = ratings.groupby("user_id").size()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(user_counts, bins=60, color="#2ca02c", edgecolor="black", alpha=0.7)
    ax.axvline(user_counts.median(), color="red", linestyle="--", linewidth=1.5, label=f"Median ({user_counts.median():.0f})")
    ax.axvline(user_counts.mean(), color="black", linestyle="-.", linewidth=1.5, label=f"Mean ({user_counts.mean():.1f})")
    ax.set_title("User Activity Distribution (Ratings per User)")
    ax.set_xlabel("Number of Ratings")
    ax.set_ylabel("Number of Users")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

    plot_path2 = output_dir / "user_activity_distribution.png"
    plt.savefig(plot_path2, dpi=300)
    plt.close()
    print(f"Saved: {plot_path2.name}")

    # 3. Rating Score Distribution
    rating_counts = ratings["rating"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(rating_counts.index, rating_counts.values, color="#ff7f0e", edgecolor="black", width=0.6, alpha=0.85)
    
    # Add count labels above bars
    total_ratings = len(ratings)
    for bar in bars:
        h = bar.get_height()
        pct = (h / total_ratings) * 100
        ax.annotate(
            f"{h:,}\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title("Rating Score Distribution")
    ax.set_xlabel("Rating (1-5)")
    ax.set_ylabel("Count")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_ylim(0, rating_counts.max() * 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    plot_path3 = output_dir / "rating_distribution.png"
    plt.savefig(plot_path3, dpi=300)
    plt.close()
    print(f"Saved: {plot_path3.name}")

    # 4. Temporal Activity Distribution
    # Resample ratings count on monthly basis
    temporal_series = ratings.set_index("datetime").resample("ME").size()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(temporal_series.index, temporal_series.values, color="#9467bd", linewidth=2, marker="o", markersize=3)
    ax.fill_between(temporal_series.index, temporal_series.values, color="#9467bd", alpha=0.25)
    ax.set_title("Temporal Activity Distribution (Monthly Rating Volume)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Ratings per Month")
    ax.grid(True, linestyle="--", alpha=0.6)

    plot_path4 = output_dir / "temporal_activity.png"
    plt.savefig(plot_path4, dpi=300)
    plt.close()
    print(f"Saved: {plot_path4.name}")


def run_data_analysis(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    users: pd.DataFrame,
    output_dir: Path | str,
) -> None:
    """Master function to execute the statistical summary and plot generations."""
    print("--- Computing Dataset Statistics ---")
    compute_statistics(ratings, movies, users, output_dir)

    print("--- Generating Characterization Plots ---")
    generate_plots(ratings, output_dir)

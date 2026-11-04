from pathlib import Path
import numpy as np
import pandas as pd

def filter_data(ratings: pd.DataFrame, min_item_ratings: int = 5, min_user_ratings: int = 20,) -> pd.DataFrame:
    """Filters low-activity items from ratings.
    
    Reports the exact threshold, rows removed, and items removeds.
    """
    initial_rows = len(ratings)
    initial_items = ratings["item_id"].nunique()
    initial_users = ratings["user_id"].nunique()

    # Calculate item counts
    item_counts = ratings["item_id"].value_counts()
    valid_items = item_counts[item_counts >= min_item_ratings].index

    # Calculate user counts (the dataset is already 20-core, which is enough)
    user_counts = ratings["user_id"].value_counts()
    valid_users = user_counts[user_counts >= min_user_ratings].index

    # Filter
    filtered = ratings[
        ratings["item_id"].isin(valid_items) & ratings["user_id"].isin(valid_users)
    ].copy()

    removed_rows = initial_rows - len(filtered)
    removed_items = initial_items - filtered["item_id"].nunique()
    removed_users = initial_users - filtered["user_id"].nunique()

    print("--- Data Filtering Report ---")
    print(f"  - Item threshold: >= {min_item_ratings} ratings per item")
    print(f"  - User threshold: >= {min_user_ratings} ratings per user")
    print(f"  - Items removed: {removed_items:,} (from {initial_items:,} to {filtered['item_id'].nunique():,})")
    print(f"  - Users removed: {removed_users:,} (from {initial_users:,} to {filtered['user_id'].nunique():,})")
    print(f"  - Interactions removed: {removed_rows:,} ({removed_rows / initial_rows * 100:.3f}% of total)")
    print(f"  - Preserved interactions: {len(filtered):,} ({len(filtered) / initial_rows * 100:.3f}%)\n")

    return filtered


def per_user_temporal_split(ratings: pd.DataFrame,
    train_ratio: float = 0.8, val_ratio: float = 0.1,) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits interactions chronologically per user into Train, Validation, and Test sets.
    
    - First 80% of interactions by timestamp -> Train
    - Next 10% of interactions by timestamp  -> Validation (hyperparameter tuning)
    - Final 10% of interactions by timestamp -> Test (final unbiased evaluation)
    """
    print("--- Executing Per-User Chronological Split ---")
    # Ensure interactions are sorted by user and timestamp
    sorted_ratings = ratings.sort_values(by=["user_id", "timestamp"]).reset_index(drop=True)

    # Compute cumulative count and total count for each user
    user_counts = sorted_ratings.groupby("user_id")["timestamp"].transform("count")
    user_cum_count = sorted_ratings.groupby("user_id").cumcount() + 1

    # Split boundaries
    train_cutoff = np.floor(user_counts * train_ratio)
    val_cutoff = np.floor(user_counts * (train_ratio + val_ratio))

    train_mask = user_cum_count <= train_cutoff
    val_mask = (user_cum_count > train_cutoff) & (user_cum_count <= val_cutoff)
    test_mask = user_cum_count > val_cutoff

    train_df = sorted_ratings[train_mask].copy()
    val_df = sorted_ratings[val_mask].copy()
    test_df = sorted_ratings[test_mask].copy()

    # Verification checks
    n_total = len(sorted_ratings)
    print(f"  - Train set : {len(train_df):,} rows ({len(train_df) / n_total * 100:.2f}%) across {train_df['user_id'].nunique():,} users")
    print(f"  - Val set   : {len(val_df):,} rows ({len(val_df) / n_total * 100:.2f}%) across {val_df['user_id'].nunique():,} users")
    print(f"  - Test set  : {len(test_df):,} rows ({len(test_df) / n_total * 100:.2f}%) across {test_df['user_id'].nunique():,} users")

    # Sanity checks
    assert len(train_df) + len(val_df) + len(test_df) == n_total, "Splits do not sum to total interactions!"
    assert train_df["user_id"].nunique() == test_df["user_id"].nunique(), "Some users are missing from test set!"

    print("  - Split verification passed: 100% of users preserved in all splits.\n")

    return train_df, val_df, test_df


def run_filter_split(
    ratings: pd.DataFrame,
    min_item_ratings: int = 5,
    min_user_ratings: int = 20,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Runs data filtering and per-user temporal split pipeline."""
    filtered_ratings = filter_data(
        ratings,
        min_item_ratings=min_item_ratings,
        min_user_ratings=min_user_ratings,
    )
    train_df, val_df, test_df = per_user_temporal_split(
        filtered_ratings,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    return train_df, val_df, test_df


"""
Temporal Splitting and Filtering Protocol Theory..

Why Item Filtering (< 5 ratings):
   - Items with fewer than 5 ratings are insufficient for things like KNN.
   - In an 80/10/10 chronological split, an item with that little interactions will disappear
     completly from one of the splits.
   - Filtering items with < 5 interactions removes only 290 items and 580 interactions
     Which is less than 0.06% of the data

2. Why Per-User Chronological Split instead of Global Timestamp Cutoff:
   - Over 90.5% of all ratings were in the first year (2000), while the rest where shared 
     among the other 2 years. (Can be seen in the temporal_activity.png)
   - A single global timestamp cutoff would severely distort evaluation as most users
     who only interacted in the 2000 would have 0 interactions in the test data.
   - Doing it Per-User sorts each user interaction by timestamp. Therefore no future interactions
     will appear in the training data, and it guarantees each user will have test items.
"""
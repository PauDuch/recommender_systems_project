import numpy as np
import pandas as pd
from part2.models.base import BaseRecommender


class ContentBasedRecommender(BaseRecommender):
    """Builds 36-dim item vectors (18 genres + 18 5-year divisions)"""

    def __init__(
        self,
        movies_df: pd.DataFrame,
        use_tfidf: bool = True,
        weight_mode: str = "centered",
        year_weight: float = 0.5,
        random_seed: int = 42,
    ):
        super().__init__(random_seed=random_seed)
        self.movies_df = movies_df
        self.use_tfidf = use_tfidf
        self.weight_mode = weight_mode
        self.year_weight = year_weight
        self.item_features = None
        self.user_profiles = None

    def _l2_normalize(self, matrix: np.ndarray) -> np.ndarray:
        """Normalizes each row of a 2D matrix"""
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def _build_item_features(self) -> np.ndarray:
        """Creates 36-dim feature vectors for all catalog items."""
        n_items = len(self.item_to_idx)
        movies_indexed = self.movies_df.set_index("item_id")

        # Collect all unique genre names across the movie catalog
        genre_set = set()
        for genres_str in self.movies_df["genres"].dropna():
            for g in genres_str.split("|"):
                genre_set.add(g)

        all_genres = sorted(genre_set)
        genre_to_col = {g: idx for idx, g in enumerate(all_genres)}
        n_genres = len(all_genres)

        genre_matrix = np.zeros((n_items, n_genres), dtype=np.float32)

        # 2. Define 5-year release buckets from 1915-1919 up to 2000-2004 (18 buckets)
        min_bucket = 1915 // 5
        max_bucket = 2000 // 5
        n_year_buckets = max_bucket - min_bucket + 1  # 18
        year_matrix = np.zeros((n_items, n_year_buckets), dtype=np.float32)

        # Fill both matrices for each item
        for item_id, idx in self.item_to_idx.items():
            if item_id not in movies_indexed.index:
                continue
            row = movies_indexed.loc[item_id]

            # Multi-hot genre encoding
            for g in str(row["genres"]).split("|"):
                if g in genre_to_col:
                    genre_matrix[idx, genre_to_col[g]] = 1.0

            # 5-year bucket one-hot encoding
            year = row["release_year"]
            if pd.notna(year):
                bucket = int(year) // 5 - min_bucket
                bucket = max(0, min(n_year_buckets - 1, bucket))
                year_matrix[idx, bucket] = 1.0

        # Apply TF-IDF weighting to genres (boosts rare genres)
        if self.use_tfidf:
            tf = self._l2_normalize(genre_matrix)
            df = np.sum(genre_matrix > 0, axis=0)
            idf = np.log((1.0 + n_items) / (1.0 + df)) + 1.0
            genre_matrix = tf * idf

        # Combine 18-dim genres + 18-dim 5-year into a 36-dim vector
        combined = np.hstack([genre_matrix, self.year_weight * year_matrix])
        return self._l2_normalize(combined)

    def fit(self, train_df: pd.DataFrame, catalog_items: list[int], **kwargs):
        """Builds item feature matrix and computes the users profile vector."""
        self._build_mappings(train_df, catalog_items)
        self.item_features = self._build_item_features()

        n_users = len(self.user_to_idx)
        n_features = self.item_features.shape[1]
        self.user_profiles = np.zeros((n_users, n_features), dtype=np.float32)

        # We have 2 modes of how ratings are rated (including negative or only postives):
        if self.weight_mode == "centered":
            # Specific weights for everything: 1 -> -2, 2 -> -1, 3 -> 0, 4 -> +1, 5 -> +2
            weights = (train_df["rating"].values - 3.0).astype(np.float32)
        elif self.weight_mode == "positive_only":
            # Binary positive weights: 1.0 if rating >= 4.0 else 0.0 (4 and 5 are the same)
            weights = (train_df["rating"].values >= 4.0).astype(np.float32)
        else:
            raise ValueError(f"Unknown weight_mode: {self.weight_mode}")

        u_indices = train_df["user_id"].map(self.user_to_idx).values
        i_indices = train_df["item_id"].map(self.item_to_idx).values

        # Add each rated movie's weighted features into the user's profile vector
        weighted_items = self.item_features[i_indices] * weights[:, np.newaxis]
        np.add.at(self.user_profiles, u_indices, weighted_items)

        # Normalize user profiles to unit length
        self.user_profiles = self._l2_normalize(self.user_profiles)

    def _predict_scores(self, u_idx: int) -> np.ndarray:
        """Scores all items via cosine similarity between user profile and item features."""
        return self.user_profiles[u_idx] @ self.item_features.T

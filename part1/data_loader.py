from pathlib import Path
import re
import pandas as pd


def load_ratings(data_dir: Path | str) -> pd.DataFrame:
    """Load ratings.dat: UserID::MovieID::Rating::Timestamp.
    
    Returns DataFrame with columns:
        user_id (int)
        item_id (int)
        rating (int)
        timestamp (int)
        datetime (pd.Timestamp)
    """
    path = Path(data_dir) / "ratings.dat"
    if not path.is_file():
        raise FileNotFoundError(f"ratings file not found at: {path}")

    ratings = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["user_id", "item_id", "rating", "timestamp"],
        dtype={"user_id": int, "item_id": int, "rating": int, "timestamp": int},
    )
    # Convert Unix timestamp to readable UTC datetime for temporal analysis
    ratings["datetime"] = pd.to_datetime(ratings["timestamp"], unit="s", utc=True)
    return ratings


def load_movies(data_dir: Path | str) -> pd.DataFrame:
    """Load movies.dat: MovieID::Title::Genres.
    
    Encoding is latin-1 to handle weird characters of other languages.
    Also extracts release_year from the title string (e.g. 'Toy Story (1995)' -> 1995).

    Returns DataFrame with columns:
        item_id (int)
        title (str)
        genres (str)
        release_year (float, NaN if missing) needs to be float, as float handles numbers + Nulls
    """
    path = Path(data_dir) / "movies.dat"
    if not path.is_file():
        raise FileNotFoundError(f"movies file not found at: {path}")

    movies = pd.read_csv(
        path,
        sep="::",
        engine="python",
        encoding="latin-1",
        names=["item_id", "title", "genres"],
        dtype={"item_id": int, "title": str, "genres": str},
    )

    # Extract 4-digit release year from parentheses at the end of title: '(YYYY)'
    extracted_year = movies["title"].str.extract(r"\((\d{4})\)$")[0]
    movies["release_year"] = pd.to_numeric(extracted_year, errors="coerce")

    return movies


def load_users(data_dir: Path | str) -> pd.DataFrame:
    """Load users.dat: UserID::Gender::Age::Occupation::Zip-code.
    
    Returns DataFrame with columns:
        user_id (int)
        gender (str)  Either M or F (could be 0 or 1 in the future)
        age (int)
        occupation (int)
        zip_code (str)
    """
    path = Path(data_dir) / "users.dat"
    if not path.is_file():
        raise FileNotFoundError(f"users file not found at: {path}")

    users = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["user_id", "gender", "age", "occupation", "zip_code"],
        dtype={"user_id": int, "gender": str, "age": int, "occupation": int, "zip_code": str},
    )
    return users


def load_all(data_dir: Path | str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_path = Path(data_dir)
    ratings = load_ratings(data_path)
    movies = load_movies(data_path)
    users = load_users(data_path)

    print(f"Data successfully loaded from {data_path.resolve()}:")
    print(f"  - Ratings : {len(ratings):,} rows across {ratings['user_id'].nunique():,} users and {ratings['item_id'].nunique():,} items")
    print(f"  - Movies  : {len(movies):,} rows")
    print(f"  - Users   : {len(users):,} rows")

    return ratings, movies, users

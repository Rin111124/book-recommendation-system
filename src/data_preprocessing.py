from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import DataConfig, ModelConfig, data_config, model_config


logger = logging.getLogger(__name__)


@dataclass
class RatingsSplit:
    train: pd.DataFrame
    test: pd.DataFrame


def load_datasets(cfg: DataConfig = data_config) -> Dict[str, pd.DataFrame]:
    """Load CSV files into memory."""
    logger.info("Loading datasets from %s", cfg.raw_dir)
    books = pd.read_csv(cfg.books_file, low_memory=False)
    users = pd.read_csv(cfg.users_file, low_memory=False)
    ratings = pd.read_csv(cfg.ratings_file, low_memory=False)
    return {"books": books, "users": users, "ratings": ratings}


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {c: c.strip().lower().replace("-", "_") for c in df.columns}
    return df.rename(columns=renamed)


def clean_data(
    books: pd.DataFrame,
    users: pd.DataFrame,
    ratings: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply light cleaning rules across datasets."""
    books = normalize_column_names(books).drop_duplicates(subset=["isbn"])
    rename_books = {}
    if "book_title" in books.columns and "title" not in books.columns:
        rename_books["book_title"] = "title"
    if "book_author" in books.columns and "author" not in books.columns:
        rename_books["book_author"] = "author"
    if rename_books:
        books = books.rename(columns=rename_books)
    users = normalize_column_names(users).drop_duplicates(subset=["user_id"])
    ratings = normalize_column_names(ratings)
    if "book_rating" in ratings.columns and "rating" not in ratings.columns:
        ratings = ratings.rename(columns={"book_rating": "rating"})
    ratings = ratings[ratings["rating"] > 0]  # drop implicit zeros
    ratings = ratings.dropna(subset=["user_id", "isbn"])
    return books, users, ratings


def filter_interactions(
    ratings: pd.DataFrame,
    cfg: ModelConfig = model_config,
) -> pd.DataFrame:
    """Keep only frequent users/items to reduce sparsity."""
    user_counts = ratings["user_id"].value_counts()
    item_counts = ratings["isbn"].value_counts()
    filtered = ratings[
        ratings["user_id"].isin(user_counts[user_counts >= cfg.min_interactions_user].index)
        & ratings["isbn"].isin(item_counts[item_counts >= cfg.min_interactions_item].index)
    ]
    logger.info(
        "Filtered ratings from %d to %d rows",
        len(ratings),
        len(filtered),
    )
    return filtered


def split_ratings(
    ratings: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> RatingsSplit:
    """Split ratings into train/test while stratifying by user."""
    stratify = ratings["user_id"] if ratings["user_id"].nunique() > 1 else None
    train_df, test_df = train_test_split(
        ratings,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    return RatingsSplit(train=train_df.reset_index(drop=True), test=test_df.reset_index(drop=True))


def build_user_item_matrix(
    ratings: pd.DataFrame,
    center: bool = True,
) -> pd.DataFrame:
    """Create a dense user-item matrix. Keep for smaller experiments."""
    matrix = ratings.pivot_table(index="user_id", columns="isbn", values="rating")
    if center:
        matrix = matrix.sub(matrix.mean(axis=1), axis=0).fillna(0.0)
    else:
        matrix = matrix.fillna(0.0)
    return matrix


def describe_ratings(ratings: pd.DataFrame) -> pd.Series:
    """Return descriptive statistics for logging/reporting."""
    stats = ratings["rating"].describe()
    logger.info("Ratings describe: \n%s", stats)
    return stats


def save_processed_data(
    books: pd.DataFrame,
    users: pd.DataFrame,
    ratings: pd.DataFrame,
    cfg: DataConfig = data_config,
) -> None:
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    books.to_csv(cfg.processed_dir / "books_processed.csv", index=False)
    users.to_csv(cfg.processed_dir / "users_processed.csv", index=False)
    ratings.to_csv(cfg.processed_dir / "ratings_processed.csv", index=False)


def preprocess_pipeline(cfg: DataConfig = data_config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = load_datasets(cfg)
    books, users, ratings = clean_data(data["books"], data["users"], data["ratings"])
    ratings = filter_interactions(ratings)
    save_processed_data(books, users, ratings, cfg)
    return books, users, ratings

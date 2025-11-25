from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import pandas as pd

from config import web_config
from src import collaborative_filtering, data_preprocessing, hybrid_model, matrix_factorization


logger = logging.getLogger(__name__)


@dataclass
class RecommenderBundle:
    books: pd.DataFrame
    users: pd.DataFrame
    ratings: pd.DataFrame
    user_cf: collaborative_filtering.UserBasedCF
    item_cf: collaborative_filtering.ItemBasedCF
    svd: matrix_factorization.SVDRecommender
    hybrid: hybrid_model.HybridRecommender


def _prepare_data() -> Optional[RecommenderBundle]:
    try:
        books, users, ratings = data_preprocessing.preprocess_pipeline()
    except FileNotFoundError as exc:
        logger.warning("Dataset not found: %s", exc)
        return None
    if ratings.empty:
        logger.warning("Ratings dataframe is empty. Ensure raw data resides in data/raw.")
        return None
    user_cf = collaborative_filtering.UserBasedCF().fit(ratings)
    item_cf = collaborative_filtering.ItemBasedCF().fit(ratings)
    svd_model = matrix_factorization.SVDRecommender().fit(ratings)
    hybrid_model_obj = hybrid_model.HybridRecommender(user_cf, item_cf, svd_model)
    return RecommenderBundle(
        books=books,
        users=users,
        ratings=ratings,
        user_cf=user_cf,
        item_cf=item_cf,
        svd=svd_model,
        hybrid=hybrid_model_obj,
    )


@lru_cache(maxsize=1)
def get_recommender_bundle() -> Optional[RecommenderBundle]:
    return _prepare_data()


def candidate_items(bundle: RecommenderBundle, limit: int = 500) -> list[str]:
    return bundle.books["isbn"].tolist()[:limit]

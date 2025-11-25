from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from config import ModelConfig, model_config


def _ensure_minimum_interactions(matrix: pd.DataFrame, axis: int, min_count: int) -> pd.DataFrame:
    if axis == 0:
        mask = (matrix != 0).sum(axis=1) >= min_count
        return matrix.loc[mask]
    mask = (matrix != 0).sum(axis=0) >= min_count
    return matrix.loc[:, mask]


@dataclass
class Recommendation:
    item_id: str
    score: float
    metadata: Optional[Dict[str, str]] = None


class BaseCF:
    def __init__(self, k: int, min_interactions: int, similarity: str = "cosine"):
        self.k = k
        self.min_interactions = min_interactions
        self.similarity_metric = similarity
        self.matrix: Optional[pd.DataFrame] = None
        self.similarity_matrix: Optional[pd.DataFrame] = None

    def _validate_fit(self) -> None:
        if self.matrix is None or self.similarity_matrix is None:
            raise RuntimeError("Model is not fitted yet.")

    def _compute_similarity(self, matrix: pd.DataFrame) -> pd.DataFrame:
        if self.similarity_metric != "cosine":
            raise NotImplementedError("Only cosine similarity is implemented.")
        sim = cosine_similarity(matrix)
        return pd.DataFrame(sim, index=matrix.index, columns=matrix.index)


class UserBasedCF(BaseCF):
    def __init__(self, cfg: ModelConfig = model_config):
        super().__init__(k=cfg.user_based_neighbors, min_interactions=cfg.min_interactions_user)

    def fit(self, ratings: pd.DataFrame) -> "UserBasedCF":
        matrix = ratings.pivot_table(index="user_id", columns="isbn", values="rating").fillna(0.0)
        matrix = _ensure_minimum_interactions(matrix, axis=0, min_count=self.min_interactions)
        self.matrix = matrix
        self.similarity_matrix = self._compute_similarity(matrix)
        return self

    def _get_neighbors(self, user_id: int) -> pd.Series:
        self._validate_fit()
        if user_id not in self.similarity_matrix.index:
            raise KeyError(f"Unknown user_id {user_id}")
        sims = self.similarity_matrix.loc[user_id].drop(user_id, errors="ignore")
        return sims.nlargest(self.k)

    def predict(self, user_id: int, isbn: str) -> float:
        self._validate_fit()
        if isbn not in self.matrix.columns:
            return 0.0
        neighbors = self._get_neighbors(user_id)
        neighbor_ratings = self.matrix.loc[neighbors.index, isbn]
        valid = neighbor_ratings != 0
        if valid.sum() == 0:
            return float(self.matrix.loc[user_id].replace(0, np.nan).mean())
        weights = neighbors[valid]
        scores = neighbor_ratings[valid]
        denom = np.abs(weights).sum()
        return float((scores * weights).sum() / denom) if denom > 0 else float(scores.mean())

    def recommend(
        self,
        user_id: int,
        books: Optional[pd.DataFrame] = None,
        top_n: int = 10,
    ) -> List[Recommendation]:
        self._validate_fit()
        if user_id not in self.matrix.index:
            raise KeyError(f"Unknown user_id {user_id}")
        user_vector = self.matrix.loc[user_id]
        unseen = user_vector[user_vector == 0]
        predictions = []
        for isbn in unseen.index:
            score = self.predict(user_id, isbn)
            predictions.append((isbn, score))
        ranked = sorted(predictions, key=lambda x: x[1], reverse=True)[:top_n]
        recs: List[Recommendation] = []
        for isbn, score in ranked:
            meta = None
            if books is not None and "isbn" in books.columns:
                match = books.loc[books["isbn"] == isbn].head(1)
                if not match.empty:
                    meta = {
                        "title": match["title"].iloc[0],
                        "author": match.get("author", pd.Series([""])).iloc[0],
                    }
            recs.append(Recommendation(item_id=isbn, score=score, metadata=meta))
        return recs


class ItemBasedCF(BaseCF):
    def __init__(self, cfg: ModelConfig = model_config):
        super().__init__(k=cfg.item_based_neighbors, min_interactions=cfg.min_interactions_item)

    def fit(self, ratings: pd.DataFrame) -> "ItemBasedCF":
        matrix = ratings.pivot_table(index="isbn", columns="user_id", values="rating").fillna(0.0)
        matrix = _ensure_minimum_interactions(matrix, axis=1, min_count=self.min_interactions)
        self.matrix = matrix
        self.similarity_matrix = self._compute_similarity(matrix)
        return self

    def _get_neighbors(self, isbn: str) -> pd.Series:
        self._validate_fit()
        if isbn not in self.similarity_matrix.index:
            raise KeyError(f"Unknown isbn {isbn}")
        sims = self.similarity_matrix.loc[isbn].drop(isbn, errors="ignore")
        return sims.nlargest(self.k)

    def predict(self, user_id: int, isbn: str) -> float:
        self._validate_fit()
        if user_id not in self.matrix.columns:
            return 0.0
        if isbn not in self.matrix.index:
            return 0.0
        neighbors = self._get_neighbors(isbn)
        user_ratings = self.matrix.loc[neighbors.index, user_id]
        valid = user_ratings != 0
        if valid.sum() == 0:
            return float(self.matrix.loc[isbn].replace(0, np.nan).mean())
        weights = neighbors[valid]
        ratings = user_ratings[valid]
        denom = np.abs(weights).sum()
        return float((ratings * weights).sum() / denom) if denom > 0 else float(ratings.mean())

    def recommend(
        self,
        user_id: int,
        books: Optional[pd.DataFrame] = None,
        top_n: int = 10,
    ) -> List[Recommendation]:
        self._validate_fit()
        if user_id not in self.matrix.columns:
            raise KeyError(f"Unknown user_id {user_id}")
        rated_items = self.matrix.loc[:, user_id]
        unseen_items = rated_items[rated_items == 0]
        predictions = []
        for isbn in unseen_items.index:
            predictions.append((isbn, self.predict(user_id, isbn)))
        ranked = sorted(predictions, key=lambda x: x[1], reverse=True)[:top_n]
        recs: List[Recommendation] = []
        for isbn, score in ranked:
            meta = None
            if books is not None:
                match = books.loc[books["isbn"] == isbn].head(1)
                if not match.empty:
                    meta = {
                        "title": match["title"].iloc[0],
                        "author": match.get("author", pd.Series([""])).iloc[0],
                    }
            recs.append(Recommendation(item_id=isbn, score=score, metadata=meta))
        return recs

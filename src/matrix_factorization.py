from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.accuracy import mae, rmse
from surprise.model_selection import KFold

from config import ModelConfig, model_config


@dataclass
class SVDEvaluation:
    rmse: float
    mae: float


class SVDRecommender:
    def __init__(self, cfg: ModelConfig = model_config):
        self.cfg = cfg
        self.reader = Reader(rating_scale=(0, 10))
        self.algorithm = SVD(
            n_factors=cfg.svd_latent_factors,
            lr_all=cfg.svd_learning_rate,
            reg_all=cfg.svd_reg,
            n_epochs=cfg.svd_epochs,
            random_state=42,
        )
        self.trainset = None

    def _build_dataset(self, ratings: pd.DataFrame) -> Dataset:
        subset = ratings[["user_id", "isbn", "rating"]]
        rating_min = float(subset["rating"].min())
        rating_max = float(subset["rating"].max())
        self.reader = Reader(rating_scale=(rating_min, rating_max))
        return Dataset.load_from_df(subset, self.reader)

    def fit(self, ratings: pd.DataFrame) -> "SVDRecommender":
        data = self._build_dataset(ratings)
        self.trainset = data.build_full_trainset()
        self.algorithm.fit(self.trainset)
        return self

    def predict(self, user_id: int, isbn: str) -> float:
        if self.trainset is None:
            raise RuntimeError("Model has not been fitted.")
        return float(self.algorithm.predict(user_id, isbn).est)

    def recommend(
        self,
        user_id: int,
        candidate_items: Iterable[str],
        top_n: int = 10,
        books: Optional[pd.DataFrame] = None,
    ) -> List[Dict[str, object]]:
        scored = [(isbn, self.predict(user_id, isbn)) for isbn in candidate_items]
        ranked = sorted(scored, key=lambda x: x[1], reverse=True)[:top_n]
        recommendations: List[Dict[str, object]] = []
        for isbn, score in ranked:
            meta = None
            if books is not None:
                book = books.loc[books["isbn"] == isbn].head(1)
                if not book.empty:
                    meta = {
                        "title": book["title"].iloc[0],
                        "author": book.get("author", pd.Series([""])).iloc[0],
                    }
            recommendations.append({"isbn": isbn, "score": score, "metadata": meta})
        return recommendations

    def evaluate(self, ratings: pd.DataFrame, folds: int = 3) -> SVDEvaluation:
        data = self._build_dataset(ratings)
        kf = KFold(n_splits=folds, random_state=42, shuffle=True)
        rmse_scores: List[float] = []
        mae_scores: List[float] = []
        for trainset, testset in kf.split(data):
            algo = SVD(
                n_factors=self.cfg.svd_latent_factors,
                lr_all=self.cfg.svd_learning_rate,
                reg_all=self.cfg.svd_reg,
                n_epochs=self.cfg.svd_epochs,
                random_state=42,
            )
            algo.fit(trainset)
            predictions = algo.test(testset)
            rmse_scores.append(rmse(predictions, verbose=False))
            mae_scores.append(mae(predictions, verbose=False))
        return SVDEvaluation(rmse=float(sum(rmse_scores) / folds), mae=float(sum(mae_scores) / folds))

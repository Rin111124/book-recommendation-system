from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import pandas as pd

from config import ModelConfig, model_config
from .collaborative_filtering import ItemBasedCF, Recommendation, UserBasedCF
from .matrix_factorization import SVDRecommender


@dataclass
class HybridWeights:
    user_cf: float = 0.4
    item_cf: float = 0.3
    svd: float = 0.3

    def normalize(self) -> "HybridWeights":
        total = self.user_cf + self.item_cf + self.svd
        if total == 0:
            return self
        return HybridWeights(
            user_cf=self.user_cf / total,
            item_cf=self.item_cf / total,
            svd=self.svd / total,
        )


class HybridRecommender:
    def __init__(
        self,
        user_cf: UserBasedCF,
        item_cf: ItemBasedCF,
        svd: SVDRecommender,
        weights: Optional[HybridWeights] = None,
        min_user_interactions: Optional[int] = None,
    ):
        self.user_cf = user_cf
        self.item_cf = item_cf
        self.svd = svd
        self.weights = (weights or HybridWeights()).normalize()
        self.min_user_interactions = min_user_interactions or model_config.min_interactions_user

    def recommend(
        self,
        user_id: int,
        candidate_items: Iterable[str],
        books: Optional[pd.DataFrame] = None,
        top_n: int = 10,
        ratings_history: Optional[pd.DataFrame] = None,
    ) -> List[Dict[str, object]]:
        available_scores: Dict[str, Dict[str, float]] = {}
        items = list(candidate_items)

        def update_scores(source: str, scores: Dict[str, float]) -> None:
            for isbn, value in scores.items():
                available_scores.setdefault(isbn, {})[source] = value

        try:
            user_preds = {rec.item_id: rec.score for rec in self.user_cf.recommend(user_id, books=None, top_n=len(items))}
            update_scores("user_cf", {k: v for k, v in user_preds.items() if k in items})
        except Exception:
            pass

        try:
            item_preds = {
                rec.item_id: rec.score
                for rec in self.item_cf.recommend(user_id, books=None, top_n=len(items))
            }
            update_scores("item_cf", {k: v for k, v in item_preds.items() if k in items})
        except Exception:
            pass

        try:
            svd_scores = {isbn: self.svd.predict(user_id, isbn) for isbn in items}
            update_scores("svd", svd_scores)
        except Exception:
            pass

        final_scores = []
        for isbn in items:
            component_scores = available_scores.get(isbn, {})
            if not component_scores:
                continue
            total_weight = 0.0
            blended_score = 0.0
            if "user_cf" in component_scores:
                blended_score += component_scores["user_cf"] * self.weights.user_cf
                total_weight += self.weights.user_cf
            if "item_cf" in component_scores:
                blended_score += component_scores["item_cf"] * self.weights.item_cf
                total_weight += self.weights.item_cf
            if "svd" in component_scores:
                blended_score += component_scores["svd"] * self.weights.svd
                total_weight += self.weights.svd
            if total_weight == 0:
                continue
            final_scores.append((isbn, blended_score / total_weight))

        ranked = sorted(final_scores, key=lambda x: x[1], reverse=True)[:top_n]
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

    def cold_start_strategy(self, user_id: int, ratings_history: pd.DataFrame) -> str:
        """Simple heuristic for switching hybrid logic."""
        user_interactions = ratings_history.loc[ratings_history["user_id"] == user_id]
        if len(user_interactions) < self.min_user_interactions:
            return "svd"
        return "weighted"

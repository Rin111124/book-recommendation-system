from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def rmse_score(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae_score(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def precision_at_k(
    recommendations: Dict[int, Sequence[str]],
    ground_truth: Dict[int, Sequence[str]],
    k: int = 10,
) -> float:
    precisions = []
    for user, recs in recommendations.items():
        truth = set(ground_truth.get(user, []))
        if not truth:
            continue
        top_k = recs[:k]
        hits = len([item for item in top_k if item in truth])
        precisions.append(hits / min(k, len(top_k)))
    return float(np.mean(precisions)) if precisions else 0.0


def recall_at_k(
    recommendations: Dict[int, Sequence[str]],
    ground_truth: Dict[int, Sequence[str]],
    k: int = 10,
) -> float:
    recalls = []
    for user, recs in recommendations.items():
        truth = set(ground_truth.get(user, []))
        if not truth:
            continue
        top_k = recs[:k]
        hits = len([item for item in top_k if item in truth])
        recalls.append(hits / len(truth))
    return float(np.mean(recalls)) if recalls else 0.0


def ndcg_at_k(
    recommendations: Dict[int, Sequence[str]],
    ground_truth_relevance: Dict[int, Dict[str, float]],
    k: int = 10,
) -> float:
    def dcg(scores: List[float]) -> float:
        return float(sum(score / np.log2(idx + 2) for idx, score in enumerate(scores)))

    ndcgs = []
    for user, recs in recommendations.items():
        rel_dict = ground_truth_relevance.get(user, {})
        gains = [rel_dict.get(item, 0.0) for item in recs[:k]]
        actual_dcg = dcg(gains)
        ideal_gains = sorted(rel_dict.values(), reverse=True)[:k]
        ideal_dcg = dcg(ideal_gains)
        if ideal_dcg == 0:
            continue
        ndcgs.append(actual_dcg / ideal_dcg)
    return float(np.mean(ndcgs)) if ndcgs else 0.0


def catalog_coverage(recommendations: Dict[int, Sequence[str]], all_items: Iterable[str]) -> float:
    recommended_items = set(item for recs in recommendations.values() for item in recs)
    catalog = set(all_items)
    return len(recommended_items) / len(catalog) if catalog else 0.0


def diversity_score(
    recommendations: Dict[int, Sequence[str]],
    item_embeddings: pd.DataFrame,
) -> float:
    """Compute average pairwise distance between recommended items."""
    distances = []
    for recs in recommendations.values():
        embeddings = item_embeddings.reindex(recs).dropna()
        vectors = embeddings.to_numpy()
        if len(vectors) < 2:
            continue
        norms = np.linalg.norm(vectors[:, None, :] - vectors[None, :, :], axis=-1)
        upper = norms[np.triu_indices(len(vectors), k=1)]
        if upper.size:
            distances.append(float(np.mean(upper)))
    return float(np.mean(distances)) if distances else 0.0

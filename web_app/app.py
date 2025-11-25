from __future__ import annotations

from typing import List

from flask import Flask, jsonify, render_template, request

from config import web_config
from .utils import RecommenderBundle, candidate_items, get_recommender_bundle


app = Flask(__name__)


def _serialize_recommendations(recommendations) -> List[dict]:
    serialized = []
    for rec in recommendations:
        if isinstance(rec, dict):
            serialized.append(rec)
        else:
            serialized.append(
                {
                    "isbn": getattr(rec, "item_id", ""),
                    "score": getattr(rec, "score", 0.0),
                    "metadata": rec.metadata if hasattr(rec, "metadata") else None,
                }
            )
    return serialized


def _generate(bundle: RecommenderBundle, user_id: int, algorithm: str, limit: int) -> List[dict]:
    if algorithm == "user_cf":
        recs = bundle.user_cf.recommend(user_id=user_id, books=bundle.books, top_n=limit)
        return _serialize_recommendations(recs)
    if algorithm == "item_cf":
        recs = bundle.item_cf.recommend(user_id=user_id, books=bundle.books, top_n=limit)
        return _serialize_recommendations(recs)
    if algorithm == "svd":
        candidates = candidate_items(bundle)
        recs = bundle.svd.recommend(user_id=user_id, candidate_items=candidates, books=bundle.books, top_n=limit)
        return recs
    candidates = candidate_items(bundle)
    recs = bundle.hybrid.recommend(
        user_id=user_id,
        candidate_items=candidates,
        books=bundle.books,
        top_n=limit,
        ratings_history=bundle.ratings,
    )
    return recs


@app.route("/", methods=["GET", "POST"])
def index():
    bundle = get_recommender_bundle()
    ready = bundle is not None
    recommendations: List[dict] = []
    error = None
    selected_algorithm = request.form.get("algorithm", "hybrid")
    user_id = request.form.get("user_id", str(web_config.default_user_id))

    if request.method == "POST":
        if not ready:
            error = "Dataset not found. Please place the Kaggle CSV files under data/raw."
        else:
            try:
                recommendations = _generate(bundle, int(user_id), selected_algorithm, web_config.recommendations_limit)  # type: ignore[arg-type]
            except Exception as exc:
                error = str(exc)

    return render_template(
        "index.html",
        ready=ready,
        recommendations=recommendations,
        error=error,
        selected_algorithm=selected_algorithm,
        default_user=user_id,
    )


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    bundle = get_recommender_bundle()
    if bundle is None:
        return jsonify({"error": "Dataset missing. Upload data to data/raw/ and refresh."}), 400
    payload = request.get_json(force=True)
    user_id = int(payload.get("user_id", web_config.default_user_id))
    algorithm = payload.get("algorithm", "hybrid")
    limit = int(payload.get("limit", web_config.recommendations_limit))
    recommendations = _generate(bundle, user_id, algorithm, limit)
    return jsonify({"recommendations": recommendations})


if __name__ == "__main__":
    app.run(debug=True)

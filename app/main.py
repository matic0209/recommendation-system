"""FastAPI service exposing dataset detail recommendations."""
from __future__ import annotations

import json
import logging
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from config.settings import DATA_DIR, MODELS_DIR

LOGGER = logging.getLogger(__name__)
app = FastAPI(title="Dataset Recommendation API")


class RecommendationItem(BaseModel):
    dataset_id: int
    title: Optional[str]
    price: Optional[float]
    cover_image: Optional[str]
    score: float
    reason: str


class RecommendationResponse(BaseModel):
    dataset_id: int
    recommendations: List[RecommendationItem]


class SimilarResponse(BaseModel):
    dataset_id: int
    similar_items: List[RecommendationItem]


def _load_pickle(path: Path) -> Dict[int, Dict[int, float]]:
    if not path.exists():
        LOGGER.warning("Model file missing: %s", path)
        return {}
    with open(path, "rb") as stream:
        return pickle.load(stream)


def _load_popular(path: Path) -> List[int]:
    if not path.exists():
        LOGGER.warning("Popular list missing: %s", path)
        return []
    return json.loads(path.read_text())


def _parse_tags(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [tag.strip().lower() for tag in str(raw).split(";") if tag.strip()]


def _load_dataset_metadata() -> Tuple[Dict[int, Dict[str, Optional[str]]], Dict[int, List[str]]]:
    meta_path = DATA_DIR / "processed" / "dataset_features.parquet"
    if not meta_path.exists():
        LOGGER.warning("Dataset feature file missing: %s", meta_path)
        return {}, {}
    frame = pd.read_parquet(meta_path)
    rename_map = {
        "dataset_id": "dataset_id",
        "dataset_name": "title",
        "price": "price",
    }
    frame = frame.rename(columns=rename_map)
    metadata: Dict[int, Dict[str, Optional[str]]] = {}
    dataset_tags: Dict[int, List[str]] = {}
    for row in frame.to_dict(orient="records"):
        dataset_id = int(row.get("dataset_id"))
        metadata[dataset_id] = {
            "title": row.get("title"),
            "price": row.get("price"),
            "cover_image": row.get("cover_image"),
        }
        dataset_tags[dataset_id] = _parse_tags(row.get("tag"))
    return metadata, dataset_tags


def _load_user_history() -> Dict[int, List[Dict[str, float]]]:
    interactions_path = DATA_DIR / "processed" / "interactions.parquet"
    if not interactions_path.exists():
        LOGGER.warning("Interactions file missing: %s", interactions_path)
        return {}
    frame = pd.read_parquet(interactions_path)
    if frame.empty:
        return {}
    frame["last_event_time"] = pd.to_datetime(frame["last_event_time"], errors="coerce")
    history: Dict[int, List[Dict[str, float]]] = {}
    for user_id, group in frame.groupby("user_id"):
        group = group.sort_values("last_event_time", ascending=False)
        total = group["weight"].sum()
        normalized = group["weight"] / total if total else group["weight"]
        records = []
        for row in group.assign(norm_weight=normalized).to_dict(orient="records"):
            records.append(
                {
                    "dataset_id": int(row["dataset_id"]),
                    "weight": float(row["norm_weight"]),
                    "last_event_time": row["last_event_time"],
                }
            )
        history[int(user_id)] = records
    return history


def _load_user_profile() -> Dict[int, Dict[str, Optional[str]]]:
    profile_path = DATA_DIR / "processed" / "user_profile.parquet"
    if not profile_path.exists():
        LOGGER.warning("User profile file missing: %s", profile_path)
        return {}
    frame = pd.read_parquet(profile_path)
    if frame.empty:
        return {}
    profiles: Dict[int, Dict[str, Optional[str]]] = {}
    for row in frame.to_dict(orient="records"):
        user_id = int(row.get("user_id"))
        profiles[user_id] = {
            "company_name": row.get("company_name"),
            "province": row.get("province"),
            "city": row.get("city"),
            "is_consumption": row.get("is_consumption"),
        }
    return profiles


def _build_user_tag_preferences(
    user_history: Dict[int, List[Dict[str, float]]],
    dataset_tags: Dict[int, List[str]],
) -> Dict[int, Dict[str, float]]:
    preferences: Dict[int, Dict[str, float]] = {}
    for user_id, records in user_history.items():
        counter: Counter = Counter()
        for record in records:
            tags = dataset_tags.get(record["dataset_id"], [])
            if not tags:
                continue
            weight = float(record["weight"])
            for tag in tags:
                counter[tag] += weight
        total = sum(counter.values())
        if total > 0:
            preferences[user_id] = {tag: weight / total for tag, weight in counter.items()}
        else:
            preferences[user_id] = {}
    return preferences


def _sorted_items(scores: Dict[int, float]) -> List[int]:
    return [item for item, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]


def _build_response_items(
    candidate_scores: Dict[int, float],
    reasons: Dict[int, str],
    limit: int,
    metadata: Dict[int, Dict[str, Optional[str]]],
) -> List[RecommendationItem]:
    result: List[RecommendationItem] = []
    for dataset_id, score in sorted(candidate_scores.items(), key=lambda kv: kv[1], reverse=True):
        info = metadata.get(dataset_id, {})
        result.append(
            RecommendationItem(
                dataset_id=dataset_id,
                title=info.get("title"),
                price=info.get("price"),
                cover_image=info.get("cover_image"),
                score=float(score),
                reason=reasons.get(dataset_id, "unknown"),
            )
        )
        if len(result) >= limit:
            break
    return result


def _combine_scores(
    target_id: int,
    behavior: Dict[int, Dict[int, float]],
    content: Dict[int, Dict[int, float]],
    popular: List[int],
    limit: int,
) -> tuple[Dict[int, float], Dict[int, str]]:
    scores: Dict[int, float] = {}
    reasons: Dict[int, str] = {}

    behavior_scores = behavior.get(target_id, {})
    for item_id, score in behavior_scores.items():
        scores[int(item_id)] = float(score)
        reasons[int(item_id)] = "behavior"

    content_scores = content.get(target_id, {})
    for item_id, score in content_scores.items():
        item_id = int(item_id)
        if item_id not in scores:
            scores[item_id] = float(score) * 0.5
            reasons[item_id] = "content"
        else:
            scores[item_id] += float(score) * 0.5

    for idx, item_id in enumerate(popular):
        if item_id == target_id or item_id in scores:
            continue
        scores[item_id] = max(0.01, 0.01 - idx * 0.0001)
        reasons[item_id] = "popular"
        if len(scores) >= limit * 3:
            break
    scores.pop(target_id, None)
    return scores, reasons


def _apply_personalization(
    user_id: Optional[int],
    scores: Dict[int, float],
    reasons: Dict[int, str],
    state,
    history_limit: int = 20,
) -> None:
    if not user_id:
        return
    user_history = state.user_history.get(int(user_id))
    if not user_history:
        return

    recent_history = user_history[:history_limit]
    history_ids = {record["dataset_id"] for record in recent_history}
    for dataset_id in history_ids:
        scores.pop(dataset_id, None)
        reasons.pop(dataset_id, None)

    tag_pref = state.user_tag_preferences.get(int(user_id), {})

    for dataset_id in list(scores.keys()):
        boost = 0.0
        for record in recent_history:
            source_id = record["dataset_id"]
            weight = record["weight"]
            sim = state.behavior.get(source_id, {}).get(dataset_id, 0.0)
            if sim:
                boost += sim * weight * 0.5
        candidate_tags = state.dataset_tags.get(dataset_id, [])
        if candidate_tags and tag_pref:
            tag_boost = sum(tag_pref.get(tag, 0.0) for tag in candidate_tags)
            boost += tag_boost * 0.2

        if boost > 0:
            scores[dataset_id] += boost
            base_reason = reasons.get(dataset_id, "unknown")
            if "personalized" not in base_reason:
                reasons[dataset_id] = f"{base_reason}+personalized"


def _get_app_state():
    if not hasattr(app.state, "models_loaded"):
        raise RuntimeError("Models are not loaded yet.")
    return app.state


@app.on_event("startup")
def load_models() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    behavior = _load_pickle(MODELS_DIR / "item_sim_behavior.pkl")
    content = _load_pickle(MODELS_DIR / "item_sim_content.pkl")
    popular = _load_popular(MODELS_DIR / "top_items.json")
    metadata, dataset_tags = _load_dataset_metadata()
    user_history = _load_user_history()
    user_profiles = _load_user_profile()
    user_tag_preferences = _build_user_tag_preferences(user_history, dataset_tags)

    app.state.behavior = behavior
    app.state.content = content
    app.state.popular = [int(item) for item in popular]
    app.state.metadata = metadata
    app.state.dataset_tags = dataset_tags
    app.state.user_history = user_history
    app.state.user_profiles = user_profiles
    app.state.user_tag_preferences = user_tag_preferences
    app.state.personalization_history_limit = 20
    app.state.models_loaded = True
    LOGGER.info(
        "Model artifacts loaded (behavior=%d items, content=%d items, popular=%d, users=%d)",
        len(behavior),
        len(content),
        len(popular),
        len(user_history),
    )


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/similar/{dataset_id}", response_model=SimilarResponse)
def get_similar(dataset_id: int, limit: int = Query(10, ge=1, le=50)) -> SimilarResponse:
    state = _get_app_state()
    scores, reasons = _combine_scores(dataset_id, state.behavior, state.content, state.popular, limit)
    items = _build_response_items(scores, reasons, limit, state.metadata)
    if not items:
        raise HTTPException(status_code=404, detail="No similar datasets found")
    return SimilarResponse(dataset_id=dataset_id, similar_items=items[:limit])


@app.get("/recommend/detail/{dataset_id}", response_model=RecommendationResponse)
def recommend_for_detail(dataset_id: int, user_id: Optional[int] = None, limit: int = Query(10, ge=1, le=50)) -> RecommendationResponse:
    state = _get_app_state()
    scores, reasons = _combine_scores(dataset_id, state.behavior, state.content, state.popular, limit)
    _apply_personalization(user_id, scores, reasons, state, state.personalization_history_limit)
    items = _build_response_items(scores, reasons, limit, state.metadata)
    return RecommendationResponse(dataset_id=dataset_id, recommendations=items[:limit])

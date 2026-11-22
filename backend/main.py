import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.append(str(Path(__file__).resolve().parent.parent))
from recsys.models import MODELS, POSITIVE

SEED = 42
np.random.seed(SEED)

DATA = Path(__file__).resolve().parent.parent / "data_raw"

movies = pd.read_csv(DATA / "movies.csv")
ratings = pd.read_csv(DATA / "ratings.csv")
tags = pd.read_csv(DATA / "tags.csv")
links = pd.read_csv(DATA / "links.csv")

ratings["ts"] = pd.to_datetime(ratings["timestamp"], unit="s")
tags["ts"] = pd.to_datetime(tags["timestamp"], unit="s")
tags["tag"] = tags["tag"].str.lower().str.strip()
movies["genres_list"] = movies["genres"].str.split("|")

TITLE = dict(zip(movies.movieId, movies.title))
GENRES = dict(zip(movies.movieId, movies.genres))
N_CATALOG = movies.movieId.nunique()

MOVIE_TAGS = (
    tags.groupby("movieId")["tag"]
    .agg(lambda s: s.value_counts().head(3).index.tolist())
    .to_dict()
)


def decorate(recs: list):
    """Attach title / genres / tags to raw recommendations."""
    for r in recs:
        m = r["movieId"]
        r["title"] = TITLE.get(m)
        r["genres"] = GENRES.get(m)
        r["tags"] = MOVIE_TAGS.get(m, [])
        for why in r["reasons"]:
            why["title"] = TITLE.get(why["movieId"])
    return recs


# What the number next to each reason means, per model.
REASON_LABEL = {
    "popularity": "Most rated movies overall, same list for everyone",
    "item_knn": "Because you liked (similarity of audiences)",
    "user_knn": "Because people with similar taste liked it, and also liked (share of those people)",
    "mf": "Because you liked (share of the match score from that movie)",
}


serving_models = {name: cls().fit(ratings) for name, cls in MODELS.items()}


app = FastAPI(title="MovieLens Recommender")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "seed": SEED}


@app.get("/summary")
def summary():
    per_user = ratings.groupby("userId").size()
    per_movie = ratings.groupby("movieId").size()
    top = per_movie.sort_values(ascending=False).head(10)
    return {
        "n_users": ratings.userId.nunique(),
        "n_movies_rated": ratings.movieId.nunique(),
        "n_movies_catalog": N_CATALOG,
        "n_ratings": len(ratings),
        "n_tags": len(tags),
        "time_range": [str(ratings.ts.min()), str(ratings.ts.max())],
        "missing_values": {
            "ratings": int(ratings[["userId", "movieId", "rating"]].isna().sum().sum()),
            "movies": int(movies[["title", "genres"]].isna().sum().sum()),
      },
        "ratings_per_user": per_user.describe().round(2).to_dict(),
        "ratings_per_movie": per_movie.describe().round(2).to_dict(),
        "share_movies_under_5_ratings": round(float((per_movie < 5).mean()), 4),
        "rating_distribution": ratings.rating.value_counts().sort_index().to_dict(),
        "top_movies": [{"movieId": int(m), "title": TITLE.get(m), "n_ratings": int(c)} for m, c in top.items()],
        "top_tags": tags.tag.value_counts().head(15).to_dict(),
    }


@app.get("/search")
def search(q: str, limit: int = 20):
    hits = movies[movies.title.str.contains(q, case=False, regex=False)].head(limit)
    n = ratings.groupby("movieId").size()
    return [
        {"movieId": int(r.movieId), "title": r.title, "genres": r.genres, "n_ratings": int(n.get(r.movieId, 0))}
        for r in hits.itertuples()
    ]


class RatingIn(BaseModel):
    movieId: int
    rating: float


class RecommendRequest(BaseModel):
    ratings: list[RatingIn] = []      # new-user preference screen
    user_id: Optional[int] = None     # or an existing MovieLens user
    k: int = 10
    model: str = "item_knn"           # "popularity", "item_knn", "user_knn" or "mf"


@app.get("/models")
def list_models():
    return [{"id": name, "label": REASON_LABEL[name]} for name in serving_models]


@app.post("/recommend")
def recommend(req: RecommendRequest):
    model = serving_models.get(req.model)
    if model is None:
        raise HTTPException(status_code=400, detail=f"Unknown model '{req.model}'. Use one of {list(serving_models)}")

    if req.user_id is not None:
        profile = model.user_profiles.get(req.user_id, {})
        recs = model.recommend_user(req.user_id, req.k)
    else:
        profile = {r.movieId: r.rating for r in req.ratings}
        recs = model.recommend(profile, req.k)

    n_liked = sum(1 for r in profile.values() if r >= POSITIVE)
    return {
        "model": req.model,
        "reason_label": REASON_LABEL[req.model],
        "n_liked_used": n_liked,
        "recommendations": decorate(recs),
    }
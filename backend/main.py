
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

SEED = 42
np.random.seed(SEED)

POSITIVE = 4.0
TEST_FRACTION = 0.20

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


class Recommender:

    def __init__(self, df: pd.DataFrame):
        self.n_users = df.userId.nunique()

        counts = df.groupby("movieId").size().sort_values(ascending=False)
        self.pop_counts = counts.to_dict()
        self.pop_rank = counts.index.tolist()

        self.user_seen = df.groupby("userId")["movieId"].apply(set).to_dict()
        liked = df[df.rating >= POSITIVE]
        self.user_liked = liked.groupby("userId")["movieId"].apply(list).to_dict()

        self.item_ids = sorted(df.movieId.unique())
        self.item_index = {m: i for i, m in enumerate(self.item_ids)}
        user_ids = sorted(df.userId.unique())
        user_index = {u: i for i, u in enumerate(user_ids)}
        rows = liked.movieId.map(self.item_index).values
        cols = liked.userId.map(user_index).values
        mat = csr_matrix(
            (np.ones(len(liked)), (rows, cols)),
            shape=(len(self.item_ids), len(user_ids)),
        )
        self.M = normalize(mat)

    def popularity(self, seen: set, k: int = 10):
        out = []
        for m in self.pop_rank:
            if m not in seen:
                out.append({"movieId": int(m), "score": float(self.pop_counts[m]), "reasons": []})
                if len(out) == k:
                    break
        return out

    def item_knn(self, liked_ids: list, seen: set, k: int = 10):
        used = [m for m in liked_ids if m in self.item_index]
        if not used:
            return self.popularity(seen, k)

        idx = [self.item_index[m] for m in used]
        S = (self.M @ self.M[idx].T).tocsr()
        scores = np.asarray(S.sum(axis=1)).ravel()
        for m in seen:
            if m in self.item_index:
                scores[self.item_index[m]] = -np.inf

        top = np.argsort(-scores)[:k]
        out = []
        for i in top:
            if scores[i] <= 0:
                break
            sims = S.getrow(i).toarray().ravel()
            best = np.argsort(-sims)[:3]
            reasons = [
                {"movieId": int(used[j]), "title": TITLE.get(used[j]), "similarity": round(float(sims[j]), 3)}
                for j in best
                if sims[j] > 0
            ]
            out.append({"movieId": int(self.item_ids[i]), "score": float(scores[i]), "reasons": reasons})
        return out


def decorate(recs: list):
    """Attach title / genres / tags to raw recommendations."""
    for r in recs:
        m = r["movieId"]
        r["title"] = TITLE.get(m)
        r["genres"] = GENRES.get(m)
        r["tags"] = MOVIE_TAGS.get(m, [])
    return recs


serving_model = Recommender(ratings)


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
    model: str = "item_knn"           # "item_knn" or "popularity"


@app.post("/recommend")
def recommend(req: RecommendRequest):
    if req.user_id is not None:
        seen = serving_model.user_seen.get(req.user_id, set())
        liked = serving_model.user_liked.get(req.user_id, [])
    else:
        seen = {r.movieId for r in req.ratings}
        liked = [r.movieId for r in req.ratings if r.rating >= POSITIVE]

    if req.model == "popularity":
        recs = serving_model.popularity(seen, req.k)
    else:
        recs = serving_model.item_knn(liked, seen, req.k)
    return {"model": req.model, "n_liked_used": len(liked), "recommendations": decorate(recs)}
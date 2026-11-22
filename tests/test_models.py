import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from recsys.models import MODELS

RATINGS = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data_raw", "ratings.csv"))
CUTOFF = RATINGS["timestamp"].quantile(0.8)
TRAIN = RATINGS[RATINGS["timestamp"] < CUTOFF]


@pytest.fixture(scope="module", params=list(MODELS))
def model(request):
    return MODELS[request.param]().fit(TRAIN)


def test_never_recommends_seen_movies(model):
    for u in list(model.user_profiles)[:40]:
        recs = model.recommend_user(u, 10)
        ids = [r["movieId"] for r in recs]
        assert not set(ids) & set(model.user_profiles[u])
        assert len(ids) == len(set(ids))
        assert len(ids) <= 10


def test_new_user_profile_works(model):
    recs = model.recommend({260: 5.0, 1: 4.5}, 5)
    assert len(recs) == 5
    assert all(r["movieId"] not in (260, 1) for r in recs)


def test_empty_profile_falls_back_to_popularity(model):
    recs = model.recommend({}, 5)
    assert len(recs) == 5


def test_model_only_knows_training_movies(model):
    train_movies = set(TRAIN["movieId"])
    for r in model.recommend({260: 5.0}, 10):
        assert r["movieId"] in train_movies


def test_reasons_come_from_the_profile(model):
    profile = {260: 5.0, 1: 4.5, 1196: 5.0}
    for r in model.recommend(profile, 10):
        assert all(w["movieId"] in profile for w in r["reasons"])
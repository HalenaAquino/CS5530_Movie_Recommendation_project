
def quality_report(movies, ratings, tags, links):
    """Basic checks on the data, returns a dict that can be saved as json."""
    per_user = ratings.groupby("userId").size()
    per_movie = ratings.groupby("movieId").size()
    user_days = ratings.groupby("userId")["timestamp"].agg(lambda s: (s.max() - s.min()) / 86400)

    # tags on movies that the same user never rated
    rated_pairs = set(zip(ratings["userId"], ratings["movieId"]))
    tags_not_rated = 0
    for u, m in zip(tags["userId"], tags["movieId"]):
        if (u, m) not in rated_pairs:
            tags_not_rated += 1

    top_users = per_user.sort_values(ascending=False).head(int(len(per_user) * 0.1))
    top_movies = per_movie.sort_values(ascending=False).head(int(len(per_movie) * 0.01))

    report = {}
    report["row_counts"] = {"movies": len(movies), "ratings": len(ratings),
                            "tags": len(tags), "links": len(links)}

    # missing values (only columns that have some)
    missing = {}
    for name, df in [("movies", movies), ("ratings", ratings), ("tags", tags), ("links", links)]:
        counts = df.isna().sum()
        missing[name] = {col: int(n) for col, n in counts.items() if n > 0}
    report["missing_values"] = missing

    report["duplicates"] = {
        "ratings_user_movie_pairs": int(ratings.duplicated(["userId", "movieId"]).sum()),
        "tags_user_movie_tag": int(tags.duplicated(["userId", "movieId", "tag"]).sum()),
        "movie_ids_in_movies": int(movies["movieId"].duplicated().sum()),
        "movie_titles_in_movies": int(movies["title"].duplicated().sum()),
    }

    valid = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    report["invalid_rating_values"] = int((~ratings["rating"].isin(valid)).sum())

    report["referential_integrity"] = {
        "ratings_movieIds_not_in_movies": int((~ratings["movieId"].isin(movies["movieId"])).sum()),
        "tags_movieIds_not_in_movies": int((~tags["movieId"].isin(movies["movieId"])).sum()),
        "movies_with_no_ratings": int((~movies["movieId"].isin(ratings["movieId"])).sum()),
    }

    report["time_range"] = {
        "ratings_min": str(ratings["date"].min()),
        "ratings_max": str(ratings["date"].max()),
        "tags_min": str(tags["date"].min()),
        "tags_max": str(tags["date"].max()),
        "ratings_per_year": {str(y): int(n) for y, n in ratings.groupby(ratings["date"].dt.year).size().items()},
    }

    report["users"] = {
        "n": int(len(per_user)),
        "min_ratings": int(per_user.min()),
        "median_ratings": float(per_user.median()),
        "max_ratings": int(per_user.max()),
        "share_with_all_ratings_in_one_day": round(float((user_days < 1).mean()), 3),
        "share_ratings_from_top_10pct_users": round(float(top_users.sum() / per_user.sum()), 3),
    }

    report["movies"] = {
        "n_catalog": int(len(movies)),
        "n_rated": int(len(per_movie)),
        "median_ratings": float(per_movie.median()),
        "share_with_1_rating": round(float((per_movie == 1).mean()), 3),
        "share_under_5_ratings": round(float((per_movie < 5).mean()), 3),
        "share_ratings_on_top_1pct_movies": round(float(top_movies.sum() / per_movie.sum()), 3),
        "no_genres_listed": int((movies["genres"] == "(no genres listed)").sum()),
        "year_unparsed": int(movies["year"].isna().sum()),
    }

    report["ratings"] = {
        "mean": round(float(ratings["rating"].mean()), 3),
        "share_ge_4": round(float((ratings["rating"] >= 4).mean()), 3),
    }

    report["tags"] = {
        "n_applications": int(len(tags)),
        "n_users_who_tagged": int(tags["userId"].nunique()),
        "n_movies_tagged": int(tags["movieId"].nunique()),
        "n_distinct_tags_raw": int(tags["tag_raw"].nunique()),
        "n_distinct_tags_normalised": int(tags["tag"].nunique()),
        "tags_on_movie_user_did_not_rate": tags_not_rated,
        "share_tags_by_top_user": round(float(tags["userId"].value_counts(normalize=True).iloc[0]), 3),
    }

    return report

import os

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data_raw")

def load_data():
    """Returns movies, ratings, tags, links as DataFrames."""
    movies = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))
    ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))
    tags = pd.read_csv(os.path.join(DATA_DIR, "tags.csv"))
    links = pd.read_csv(os.path.join(DATA_DIR, "links.csv"))

    # timestamps are in seconds
    ratings["date"] = pd.to_datetime(ratings["timestamp"], unit="s")
    tags["date"] = pd.to_datetime(tags["timestamp"], unit="s")

    # tags are free text so make them lowercase and strip spaces
    tags["tag_raw"] = tags["tag"]
    tags["tag"] = tags["tag"].str.lower().str.strip()

    # release year is at the end of the title, like "Toy Story (1995)"
    year = movies["title"].str.extract(r"\((\d{4})\)\s*$")[0]
    movies["year"] = pd.to_numeric(year)
    movies["genre_list"] = movies["genres"].str.split("|")

    return movies, ratings, tags, links
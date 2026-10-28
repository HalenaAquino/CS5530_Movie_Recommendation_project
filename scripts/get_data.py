# Downloads the MovieLens small dataset into data_raw/ and records where it came from.
# Run from anywhere:  python scripts/get_data.py        (skips download if files exist)
#                     python scripts/get_data.py --force (download again)
import io
import json
import os
import sys
import urllib.request
import zipfile
from datetime import date

import pandas as pd

URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data_raw")
CSV_FILES = ["movies.csv", "ratings.csv", "tags.csv", "links.csv"]

# row counts from the dataset README
EXPECTED = {"movies.csv": 9742, "ratings.csv": 100836, "tags.csv": 3683}


def download():
    print("downloading", URL)
    data = urllib.request.urlopen(URL).read()
    z = zipfile.ZipFile(io.BytesIO(data))
    for name in CSV_FILES:
        with open(os.path.join(OUT_DIR, name), "wb") as f:
            f.write(z.read("ml-latest-small/" + name))
        print("saved", name)


def main():
    force = "--force" in sys.argv
    have_files = all(os.path.exists(os.path.join(OUT_DIR, f)) for f in CSV_FILES)

    source_path = os.path.join(OUT_DIR, "DATA_SOURCE.json")
    old = {}
    if os.path.exists(source_path):
        old = json.load(open(source_path))

    access_date = old.get("access_date", "unknown (run with --force)")
    if force or not have_files:
        download()
        access_date = str(date.today())
    else:
        print("files already in data_raw/, not downloading")

    # check the row counts
    rows = {}
    for f in CSV_FILES:
        rows[f] = len(pd.read_csv(os.path.join(OUT_DIR, f)))
        if f in EXPECTED and rows[f] != EXPECTED[f]:
            print("WARNING:", f, "has", rows[f], "rows, expected", EXPECTED[f])
    print("row counts:", rows)

    info = {"url": URL, "access_date": access_date, "rows": rows}
    with open(source_path, "w") as f:
        json.dump(info, f, indent=2)


if __name__ == "__main__":
    main()

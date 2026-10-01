import pandas as pd

# Loads all data into Dataframes
links = pd.read_csv('../data_raw/links.csv', encoding='utf-8')
movies = pd.read_csv('../data_raw/movies.csv', encoding='utf-8')
ratings = pd.read_csv('../data_raw/ratings.csv', encoding='utf-8')
tags = pd.read_csv('../data_raw/tags.csv', encoding='utf-8')

for movie in movies.itertuples():
    # Combines all data into single table, should be delted later - just for testing purposes
    merged = movies.join(links.set_index('movieId'), on='movieId', how='left')
    merged = merged.join(ratings.groupby('movieId').agg({'rating': 'mean'}), on='movieId', how='left')
    merged = merged.join(tags.groupby('movieId').agg({'tag': lambda x: ', '.join(x)}), on='movieId', how='left')


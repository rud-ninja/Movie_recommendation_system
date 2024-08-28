import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix


class data_preprocess():

    def __init__(self):
        self.movies = pd.read_csv('D:/movie_recommendation_system/move_rec/movies.csv')
        self.ratings = pd.read_csv('D:/movie_recommendation_system/move_rec/ratings.csv').drop(columns=['timestamp'])
        self.movies['old_id'] = self.movies['movieId']
        self.movies['movieId'] = self.movies.index


    def transform(self):
        id_mapper = {k: v for k, v in zip(self.movies['old_id'], self.movies['movieId'])}
        self.ratings['movieId'] = self.ratings['movieId'].map(id_mapper)
        self.ratings['userId'] = self.ratings['userId'] - 1

        movie_dict = {k: v for k, v in zip(self.movies['movieId'], self.movies['title'])}

        self.ratings['userId'] = self.ratings['userId'].astype('category')
        self.ratings['movieId'] = self.ratings['movieId'].astype('category')

        arr = csr_matrix((self.ratings['rating'].values, (self.ratings['userId'].cat.codes, self.ratings['movieId'].cat.codes)))

        coom = arr.tocoo()
        row_indices = coom.row
        column_indices = coom.col
        values = coom.data

        # Calculate sum and count of values per row
        row_sums = np.bincount(row_indices, weights=values)
        row_counts = np.bincount(row_indices)

        # Compute the mean for each row
        row_means = row_sums / row_counts

        # Normalize values by subtracting the mean
        normalized_values = values - row_means[row_indices]

        arr = csr_matrix((normalized_values, (row_indices, column_indices)))

        return movie_dict, arr
    

    def get_genres(self, k):
        genres = self.movies['genres'].loc[k]
        
        return genres
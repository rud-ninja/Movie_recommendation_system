import re
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer



class create_repo():

    def __init__(self):
        base_folder = r"C:/Users/User/Desktop/movie_recommendation_system/move_rec"
        self.movies = pd.read_csv(os.path.join(base_folder, "movies.csv"))
        self.ratings = pd.read_csv(os.path.join(base_folder, "ratings.csv"))

    def clean(self, text):
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text

    def process_n_vectorize(self):
        self.movies['clean_title'] = self.movies['title'].apply(lambda x: self.clean(x))

        # Fit TF-IDF vectorizer for embedding words for similarity check
        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(self.movies['clean_title'])

        return self.movies, self.ratings, vectorizer, tfidf_matrix
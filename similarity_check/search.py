import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from process_data import create_repo


class search():

    def __init__(self):
        data_obj = create_repo()
        self.movies, _, self.vectorizer, self.tfidf_matrix = data_obj.process_n_vectorize()


    # Search movie titles similar to queried text using Cosine similarity
    def query_search(self, query):
        query_vector = self.vectorizer.transform([query])
        cosine_similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        indices = np.argpartition(cosine_similarities, -10)[-10:]
        results = self.movies.iloc[indices][::-1]

        return results
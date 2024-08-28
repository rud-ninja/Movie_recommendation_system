import pandas as pd
from process_data import create_repo

class recommendations():

    def __init__(self):
        data_obj = create_repo()
        self.movies, self.ratings, _, _ = data_obj.process_n_vectorize()

    def get_recommendations(self, movie_id):

        # Find users who watched the same movie and liked it (i.e. rated above 5)
        similar_users = self.ratings[(self.ratings['movieId']==movie_id) & (self.ratings['rating'] > 4)]['userId'].unique()
        # Find other movies watched by above users and rated highly
        similar_user_recs = self.ratings[(self.ratings['userId'].isin(similar_users)) & (self.ratings['rating'] > 4)]['movieId']


        # Filter out the movies that have been watched by at least 20% of the users
        similar_user_recs = similar_user_recs.value_counts() / len(similar_users)
        similar_user_recs = similar_user_recs[similar_user_recs > 0.2]


        # Now that we know how much similar users liked the recommendedations. Now we will get how all users liked the recommendations. Similar and dissimilar to user
        all_users = self.ratings[(self.ratings['movieId'].isin(similar_user_recs.index)) & (self.ratings['rating'] > 4)]
        # Normalize the counts with total number of unique uses there are
        all_user_recs = all_users['movieId'].value_counts() / len(all_users['userId'].unique())


        # Create a score metric (how much similar people like it / how much people like it in general). A Higher score suggests a more targeted recommendation
        rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)
        rec_percentages.columns = ['similar', 'all']
        rec_percentages['score'] = rec_percentages['similar'] / rec_percentages['all']
        rec_percentages = rec_percentages.sort_values('score', ascending=False)

        # Return top k recommendations
        return rec_percentages.head(5).merge(self.movies, left_index=True, right_on='movieId')[['score', 'title', 'genres']]
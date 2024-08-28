import numpy as np
import random
import pandas as pd
from data_preprocessing import data_preprocess
from tabulate import tabulate


P = np.load('D:/movie_recommendation_system/matrices/P.npy')
Q = np.load('D:/movie_recommendation_system/matrices/Q.npy')


data = data_preprocess()
movie_dict, arr = data.transform()


user = random.randint(0, arr.shape[0])
user_all_movies = sorted({k: v for k, v in enumerate(np.array(arr[user, :].todense())[0]) if v>0}.items(), key=lambda x: x[1], reverse=True)
user_movies = dict(user_all_movies[:5])
user_movie_id = [x[0] for x in user_all_movies]


user_col = f"Favourites of User# {user}"
display_df = pd.DataFrame(columns=[user_col, "Recommendations"])


temp = []
for i, k in enumerate(user_movies.keys()):
    temp.append(f"{movie_dict[k]}\nGenre: {data.get_genres(k)}\n\n")

display_df[user_col] = temp



preds = P[user, :] @ Q

for k in user_movie_id:
    preds[k] = 0

preds = sorted({k: v for k, v in enumerate(preds)}.items(), key=lambda x: x[1], reverse=True)[:5]
preds = [x[0] for x in preds]


temp = []
for i, k in enumerate(preds):
    temp.append(f"{movie_dict[k]}\nGenre: {data.get_genres(k)}\n\n")

display_df["Recommendations"] = temp


print(tabulate(display_df, headers='keys', tablefmt='fancy_grid', showindex=False))
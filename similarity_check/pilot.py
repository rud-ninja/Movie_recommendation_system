from search import search
from recommend import recommendations

query = input("Enter movie name: ")

search_obj = search()
search_results = search_obj.query_search(query)

rec_obj = recommendations()
top_search = search_results.iloc[0]['movieId']
top_rec = rec_obj.get_recommendations(top_search)


print(f"\n\nIf you liked: {query}\n\nYou may also like:\n")

try:
    for index in top_rec.index:
        print(f"--> {top_rec['title'].loc[index]}")
    print("\n\n\n")

except:
    print(top_rec)
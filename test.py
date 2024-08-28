import pandas as pd

df1 = pd.DataFrame.from_dict({'A': [1], 'B': [2]})
df2 = pd.DataFrame.from_dict({'C': [3], 'B': [4]})

df = pd.concat([df1, df2], ignore_index=False)

print(df)
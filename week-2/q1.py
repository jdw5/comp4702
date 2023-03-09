import pandas as pd
df = pd.read_csv("pokemonsrt.csv")
print(f"{df.head()}\n")
print(f"{df.info()}\n")
print(f"{df.isnull().sum()}\n")

for column in df:
    print(f"{df[column].describe()}\n")
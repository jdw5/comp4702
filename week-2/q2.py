import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

df = pd.read_csv("pokemonsrt.csv")
og = df

# Remove 'type2' column as it is not quantitative data
df.drop(columns=['type2'], inplace=True)

# Replace 'weight_kg', 'percentage_male', 'height_m' with mean
cols_to_impute = ['weight_kg', 'percentage_male', 'height_m']
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')

for col in cols_to_impute:
    df[col] = imp_mean.fit_transform(df[[col]])
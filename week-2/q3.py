import pandas as pd
import numpy as np
import seaborn
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from multiprocessing import Process

df = pd.read_csv("pokemonsrt.csv")
og = df

# Remove 'type2' column as it is not quantitative data
df.drop(columns=['type2'], inplace=True)

# Replace 'weight_kg', 'percentage_male', 'height_m' with mean
cols_to_impute = ['weight_kg', 'percentage_male', 'height_m']
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')

for col in cols_to_impute:
    df[col] = imp_mean.fit_transform(df[[col]])

only_plot = ['weight_kg', 'height_m', 'percentage_male']

def show_imputed() :
    seaborn.pairplot(df[only_plot])
    plt.suptitle('Imputed')
    plt.show()

def show_original() :
    seaborn.pairplot(og[only_plot])
    plt.suptitle('Original')
    plt.show()

imp = Process(target=show_imputed)
org = Process(target=show_original)
imp.start()
org.start()
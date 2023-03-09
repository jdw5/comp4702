import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Process


# Label the columns x,y,z
df1 = pd.read_csv('w3classif.csv', header=None, names=['x', 'y', 'z'])
df2 = pd.read_csv('w3regr.csv', header=None, names=['x', 'y', 'z'])


def show_df1() :
    plt.scatter(df1['x'], df1['y'])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Scatter Plot for w3classif')
    plt.show()

def show_df2() :
    plt.scatter(df2['x'], df2['y'])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Scatter Plot for w3regr')
    plt.show()



process_df1 = Process(target=show_df1)
process_df2 = Process(target=show_df2)
process_df1.start()
process_df2.start()
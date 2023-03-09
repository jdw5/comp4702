import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Process


# Label the columns x,y,z - we are only interested in x,y
df1 = pd.read_csv('w3classif.csv', header=None, names=['x', 'y', 'z'])
df2 = pd.read_csv('w3regr.csv', header=None, names=['x', 'y', 'z'])

# Plot the scatters for x, y
def show_df1() :
    plt.scatter(df1['x'], df1['y'])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Scatter Plot for w3classif')
    plt.show()

# Plot the scatters for x, y 
def show_df2() :
    plt.scatter(df2['x'], df2['y'])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Scatter Plot for w3regr')
    plt.show()

def show_both() :
    plt.scatter(df1['x'], df1['y'])
    plt.scatter(df2['x'], df2['y'])
    plt.show()
# Plot both
process_df1 = Process(target=show_df1)
process_df2 = Process(target=show_df2)
process_both = Process(target=show_both)
process_df1.start()
process_df2.start()
process_both.start()
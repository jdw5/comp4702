import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from multiprocessing import Process



df1 = pd.read_csv('w3classif.csv', header=None, names=['x', 'y', 'z'])
df2 = pd.read_csv('w3regr.csv', header=None, names=['x', 'y', 'z'])

# df = df.sample(frac=1).reset_index(drop=True)

classif_train, classif_test = train_test_split(df1, test_size=0.3)
regr_train, regr_test = train_test_split(df2, test_size=0.3)

# Plot the scatters for x, y
def classif() :
    plt.scatter(classif_train['x'], classif_train['y'])
    plt.scatter(classif_test['x'], classif_test['y'])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Test vs Training Data Plot for Classif')
    plt.show()

process_classif_train = Process(target=classif)
process_classif_train.start()


def regr() :
    plt.scatter(regr_train['x'], regr_train['y'])
    plt.scatter(regr_test['x'], regr_test['y'])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Test vs Training Data Plot for Regresssion Set')
    plt.show()

process_regr_train = Process(target=regr)
process_regr_train.start()
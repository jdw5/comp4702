import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

k = 4
df = pd.read_csv('w3classif.csv', header=None)

# Get x values and y values
# coordinates provides array[2] of points
# output is 1 or 0
coordinates = df.iloc[:, :-1]
output = df.iloc[:, -1]

# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
# Separates the shuffled arrays passed in as params (coordinates, output) into test and training sets 
trainX, testX, trainY, testY = train_test_split(coordinates, output.values, test_size=0.3, shuffle=True)

# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
# Generate a majority vote classifier for the specified k-value, eg: k=3 means
# 3 votes are required for classification
knn = KNeighborsClassifier(n_neighbors = k)

# Train the classifier using the generated testX and testY
knn.fit(trainX, trainY)

# Predict the expected values
train_preds = knn.predict(trainX)
test_preds = knn.predict(testX)

# Find the miscalculation rate
train_loss = (1 - accuracy_score(trainY, train_preds)) * 100
test_loss = (1 - accuracy_score(testY, test_preds)) * 100
print(f'Training Loss: %1.3f%%\nTest Loss: %1.3f%%' % (train_loss, test_loss))

# Get the min, max for the coordinates to determine the range in x and y
x_min, x_max = coordinates.values[:, 0].min() - 1, coordinates.values[:, 0].max() + 1
y_min, y_max = coordinates.values[:, 1].min() - 1, coordinates.values[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision regions for the k-NN classifier together with the training and/or test data points
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(trainX.iloc[:, 0], trainX.iloc[:, 1], c = trainY, s=20, edgecolor = 'k')
plt.scatter(testX.iloc[:, 0], testX.iloc[:, 1], c = testY, s=20, edgecolor = 'k', marker='s')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Classifier Decision Regions for k = %d' % k)
plt.show()










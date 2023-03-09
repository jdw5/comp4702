import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df = pd.read_csv("w3regr.csv")

# Split the data into features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

# Create the k-NN regressor
reg = KNeighborsRegressor(n_neighbors=3)

# Fit the regressor to the training data
reg.fit(X_train, y_train)

# Make predictions on the training and testing data
y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)

# Calculate the training and testing loss
train_loss = mean_squared_error(y_train, y_train_pred)
test_loss = mean_squared_error(y_test, y_test_pred)

print(f"Training loss: %.3f" % train_loss)
print(f"Testing loss: %.3f" % test_loss)


fig, ax = plt.subplots()
ax.scatter(X_train, y_train, color='blue', label='Training Data')
ax.scatter(X_test, y_test, color='green', label='Testing Data')
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
y_combined_pred = reg.predict(X_combined)
ax.plot(X_combined, y_combined_pred, color='red', label='Predicted Function')
ax.set_xlabel('X')
ax.set_ylabel('y')
ax.set_title('k-NN Regression (k=3)\nTraining Loss={:.2f}, Testing Loss={:.2f}'.format(train_loss, test_loss))
ax.legend()
plt.show()
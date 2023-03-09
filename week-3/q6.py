import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("w3regr.csv")
depth = 2

# Split the data into features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the decision tree regressor
reg = DecisionTreeRegressor(max_depth = depth)
depthMax = DecisionTreeRegressor()
depthFive = DecisionTreeRegressor(max_depth = 4)
depthTen = DecisionTreeRegressor(max_depth = 6)
reg.fit(X_train, y_train)
depthMax.fit(X_train, y_train)
depthFive.fit(X_train, y_train)
depthTen.fit(X_train, y_train)

# Make predictions on the training and testing data
y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)

# Calculate the training and testing loss
train_loss = mean_squared_error(y_train, y_train_pred)
test_loss = mean_squared_error(y_test, y_test_pred)

print("Training Loss (MSE):", train_loss)
print("Testing Loss (MSE):", test_loss)


plt.scatter(X_train, y_train, label='Training Data')
plt.scatter(X_test, y_test, label='Test Data')
plt.plot(X, reg.predict(X), color='red', label='Depth = 2')
plt.plot(X, depthFive.predict(X), color='green', label='Depth = 4')
plt.plot(X, depthTen.predict(X), color='black', label='Depth = 6')
plt.plot(X, depthMax.predict(X), color='gray', label='Maximum Depth')
plt.legend()
plt.title('Decision Tree Regression')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.show()
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load the data
df = pd.read_csv("w3classif.csv")

# Split the data into features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Fit the classifier to the training data
clf.fit(X_train, y_train)

# Make predictions on the training and testing data
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

# Calculate the training and testing accuracy
train_loss = 1 - accuracy_score(y_train, y_train_pred)
test_loss = 1 - accuracy_score(y_test, y_test_pred)

print("Training loss: {:.2f}%".format(train_loss * 100))
print("Testing loss: {:.2f}%".format(test_loss * 100))
# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset (replace 'dataset.csv' with your dataset)
data = pd.read_csv('dataset.csv')

# Data preprocessing: handle missing values, encode categorical variables, etc.
# For simplicity, let's assume the dataset is already preprocessed

# Splitting the dataset into features and target variable
X = data.drop('fraud_label', axis=1)  # Assuming 'fraud_label' is the target variable
y = data['fraud_label']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Making predictions on the test set
y_pred = clf.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('assignment_2_data.csv')
 
# region STEP1: binary classification using logistic regression

data['certified'] = data['certified'].replace(['yes', 'no'], [1, 0])

y1 = data['certified'].values
x1 = data[['forum.posts', 'grade', 'assignment']].values

# Split the dataset
# X_train, X_test, Y_train, Y_test = train_test_split(x1, y1, test_size=0.3, random_state=1, stratify=y1)

# # Logistic Regression with smaller C value for stronger regularization
# lr = LogisticRegression(C=1.0, random_state=1, solver='lbfgs', multi_class='ovr')
# lr.fit(X_train, Y_train)

# # Predictions
# Y_predict = lr.predict(X_test)

# # Output performance metrics
# print("Recall  :%.3f" % metrics.recall_score(Y_test, Y_predict))
# print("Precision :%.3f" % metrics.precision_score(Y_test, Y_predict))
# print("F1 Score :%.3f" % metrics.f1_score(Y_test, Y_predict))
# print("Accuracy :%.3f" % metrics.accuracy_score(Y_test, Y_predict))

# # Display confusion matrix
# conf_matrix = metrics.confusion_matrix(Y_test, Y_predict)
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()

# # Check the distribution of the certified column
# print(data['certified'].value_counts())

# # Check for feature correlation
# print(data[['forum.posts', 'grade', 'assignment']].corr())
# endregion

# region STEP 2: decision tree
# Define the target variable (y) and features (X)
# y = data['certified'].values
# X = data[['forum.posts', 'grade', 'assignment']].values

# # Split the data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# # Initialize the Decision Tree Classifier
# dt = DecisionTreeClassifier(random_state=1)

# # Train the model
# dt.fit(X_train, y_train)

# # Predict on the test set
# y_pred = dt.predict(X_test)

# # Print evaluation metrics
# print("Decision Tree Recall  :%.3f" % metrics.recall_score(y_test, y_pred))
# print("Decision Tree Precision :%.3f" % metrics.precision_score(y_test, y_pred))
# print("Decision Tree F1 Score :%.3f" % metrics.f1_score(y_test, y_pred))
# print("Decision Tree Accuracy :%.3f" % metrics.accuracy_score(y_test, y_pred))

# # Display confusion matrix
# conf_matrix = metrics.confusion_matrix(y_test, y_pred)
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()

# # Check the distribution of the 'certified' column
# print(data['certified'].value_counts())

# # Check for feature correlation (optional)
# print(data[['forum.posts', 'grade', 'assignment']].corr())
# endregion

# region STEP 3: Naive Bayes
# Define the target variable (y) and features (X)
y = data['certified'].values
X = data[['forum.posts', 'grade', 'assignment']].values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# Initialize the Naive Bayes model (GaussianNB for continuous data)
nb = GaussianNB()

# Train the model
nb.fit(X_train, y_train)

# Predict on the test set
y_pred = nb.predict(X_test)

# Print evaluation metrics
print("Naive Bayes Recall  :%.3f" % metrics.recall_score(y_test, y_pred))
print("Naive Bayes Precision :%.3f" % metrics.precision_score(y_test, y_pred))
print("Naive Bayes F1 Score :%.3f" % metrics.f1_score(y_test, y_pred))
print("Naive Bayes Accuracy :%.3f" % metrics.accuracy_score(y_test, y_pred))

# Display confusion matrix
conf_matrix = metrics.confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Naive Bayes')
plt.show()

# Check the distribution of the 'certified' column
print(data['certified'].value_counts())

# Check for feature correlation (optional)
print(data[['forum.posts', 'grade', 'assignment']].corr())
# endregion
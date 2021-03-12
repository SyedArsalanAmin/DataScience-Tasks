# imporing libraries
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
sns.set()

# loading dataset
dataset = pd.read_csv(
    "E:\\DataScience & AI\\Github_repo\\datasets\\student_scores - student_scores.csv", delimiter=",")

# Visualizing and exploring data
dataset.shape
dataset.describe()
sns.pairplot(dataset)  # to see correlation between the two features
plt.scatter(dataset["Hours"], dataset["Scores"], c='r')
plt.xlabel("Study Hours")
plt.ylabel("Percentage Obtained")
plt.title("Percentage vs Hours Studied")


dataset.shape
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values
X.shape
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)  # splitting in train and test data

model = LinearRegression()
model.fit(X_train, y_train)

model.predict(X_test)
model.predict([[9.25]])
# -------------------------------------------------------------------------------

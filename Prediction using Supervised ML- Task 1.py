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
    "E:\DataScience & AI\Github_repo\datasets\student_scores - student_scores.csv", delimiter=",")

dataset.shape

sns.pairplot(dataset)  # to see correlation between the two features
plt.scatter(dataset["Hours"], dataset["Scores"], c='r')
plt.xlabel("Study Hours")
plt.ylabel("Percentage Obtained")
plt.title("Percentage vs Hours Studied")

dataset.head()
X = dataset.iloc[:, 0].values
y = dataset.iloc[:, 1].values

X_train, X_test, y_taintrain_test_split(X, y, test_size=0.2)

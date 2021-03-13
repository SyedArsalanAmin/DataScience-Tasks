# %%markdown
# # Author: __Syed Arsalan Amin__
# ## The Sparks Foundation
# ### DataScience and Business Intelligence Internship
# Task-1: Predict the percentage of an student based on the no. of study hours.What will be predicted score if a student studies for 9.25 hrs/ day?
# ### Dataset: student_scores - student_scores.csv
# ### Download dataset from here: [student_scores.csv](http://bit.ly/w-data)

# %%codecell
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd
import seaborn as sns
sns.set()
# %%markdown
# ## Loading and Exploring dataset

# %%codecell
dataset = pd.read_csv(
    "E:\\DataScience & AI\\Github_repo\\datasets\\student_scores - student_scores.csv", delimiter=",")  # loading dataset

dataset.shape
dataset.describe()
sns.pairplot(dataset)  # to see correlation between the two features


dataset.shape
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values
X.shape
# %%markdown
# ## splitting dataset in train and test set

# %%codecell

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)  # splitting in train and test data

model = LinearRegression()  # loading model
model.fit(X_train, y_train)  # training model

y_pred = model.predict(X_test)
pd.DataFrame({'Actual_Values': y_test, 'Predicted_values': y_pred})

# %%markdown
# ### Visualizing the data and fiiting line from the trained model

# %%codecell


def plot_scatter(x, y):
    plt.scatter(dataset[x], dataset[y], c='r')
    plt.xlabel("Study Hours")
    plt.ylabel("Percentage Obtained")
    plt.title("Percentage vs Hours Studied")


def plot_line():
    # now to use the above data to draw a best-fit line
    line = model.coef_ * X_test + model.intercept_
    plt.plot(X_test, line)


plot_scatter("Hours", "Scores")
plot_line()

# %%markdown
# ## Now if student studies 9.25 hours how many marks would he gain?
# For this prediction use the above trained LinearRegression model
# %%codecell
hours = 9.25
pred_marks = model.predict([[hours]])
print(f"The student obtains {int(pred_marks[0])} marks if he studies {hours} hours.")
# %%markdown
# ## Model evaluation
# Calculating mean_absolute_error
# %%codecell
print(f"Model has {mean_absolute_error(y_test, y_pred)} of MAE.")

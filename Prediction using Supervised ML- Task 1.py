# %%markdown
# - Author: __Syed Arsalan Amin__
# - DaraScience and Business Intelligence Intern
# - The Sparks Foundation
# Task-1: Predict the percentage of an student based on the no. of study hours.What will be predicted score if a student studies for 9.25 hrs/ day?
###
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
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


dataset.shape
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values
X.shape
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)  # splitting in train and test data

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
pd.DataFrame({'Actual_Values': y_test, 'Predicted_values': y_pred})

# Visualizing the data and fiiting line from the trained model


def plot_scatter(x, y):
    plt.scatter(dataset[x], dataset[y], c='r')
    plt.xlabel("Study Hours")
    plt.ylabel("Percentage Obtained")
    plt.title("Percentage vs Hours Studied")


def plot_line():
    # now to use the above data to draw a best-fit line
    x_fit = np.linspace(1, 10, 10)
    y_fit = model.predict(x_fit[:, np.newaxis])
    return(plt.plot(x_fit, y_fit))


plot_scatter("Hours", "Scores")
plot_line()

# Now if student studies 9.25 Hours how many marks would he gain?
hours = 9.25
pred_marks = model.predict([[hours]])
print(f"The student obtains {int(pred_marks[0])} marks if he studies {hours} hours.")

# model evaluation
print(f"Model has {mean_absolute_error(y_test, y_pred)} of MAE.")
# -------------------------------------------------------------------------------

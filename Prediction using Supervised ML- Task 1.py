# imporing libraries
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
sns.set()

# loadnig dataset
dataset = pd.read_csv(
    "E:\DataScience & AI\Github_repo\datasets\student_scores - student_scores.csv", delimiter=",")

dataset.shape
dataset.head()
sns.pairplot(dataset)  # to see correlation betweent the two features

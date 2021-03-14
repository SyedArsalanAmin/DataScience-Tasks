import numpy as np
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import sklearn.metrics as accuracy_score
import pandas as pd
import seaborn as sns
sns.set()


# importing data
df = datasets.load_iris()
df.target_names  # we've 3 categories
features_set = df.data
features_set.shape
features_set[:3]


def scatter_plot(c1, c2):
    plt.scatter(features_set[:, c1], features_set[:, c2])
    plt.xlabel("Lenght")
    plt.ylabel("Width")
    plt.title("Petal anaylysis")


scatter_plot(2, 3)

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features_set)
scaled_features.shape

scaled_features[:3]  # these are the normalized feature set

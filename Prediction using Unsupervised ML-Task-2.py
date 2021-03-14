import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import sklearn.metrics as accuracy_score
import pandas as pd

# importing data
df = datasets.load_iris()
df.target
df.data[:3]
df.target_names
features_set = df.data
features_set.shape
features_set[:3]
plt.scatter(features_set[:, 2], features_set[:, 3])

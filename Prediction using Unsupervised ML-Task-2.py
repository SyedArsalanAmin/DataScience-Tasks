import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import sklearn.metrics as accuracy_score
import pandas as pd
import seaborn as sns
sns.set()


# importing data
df = pd.read_csv("E:\DataScience & AI\Github_repo\datasets\Iris.csv")
df = df.drop(columns=['Species', 'Id'])
df.head()


def scatter_plot(dataset, col1, col2):
    plt.scatter(dataset.iloc[:, col1], dataset.iloc[:, col2])
    plt.xlabel("Lenght")
    plt.ylabel("Width")
    plt.title("Petal anaylysis")


scatter_plot(df, 2, 3)

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df)
scaled_features.shape
scaled_features[:3]  # these are the normalized feature set between 0-1

# Now using kmeans to preict the number of cluster
kmeans = KMeans(n_clusters=3)
y_pred = kmeans.fit_predict(scaled_features)
y_pred  # so thses are the predicted categories of the data we provided to kmeans
df["Clusters"] = y_pred
df.head()
df.describe()  # looking into the data for insights
# for a better understadin of the data lets take a look at the correlation between different features
sns.pairplot(df)


# Making  Clusters
cluster1 = df[df['Clusters'] == 0].reset_index()
cluster1.head(3)
cluster2 = df[df['Clusters'] == 1].reset_index()
cluster2.head(3)
cluster3 = df[df['Clusters'] == 2].reset_index()
cluster3.head(3)

# Cluster centers
centers = kmeans.cluster_centers_[:3, -2:]  # cluster center for petals
centers
plt.scatter(centers[0, 0], centers[0, 1], c='r')
plt.scatter(centers[1, 0], centers[1, 1], c='b')
plt.scatter(centers[2, 0], centers[2, 1], c='y')

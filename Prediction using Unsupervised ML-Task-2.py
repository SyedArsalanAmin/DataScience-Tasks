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


scatter_plot(df, 2, 3)  # visualizing petal data
scatter_plot(df, 0, 1)  # visualizing sepal data

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df)
scaled_features.shape
scaled_features[:3]  # these are the normalized feature set between 0-1

# Now using kmeans to preict the number of cluster
kmeans = KMeans(n_clusters=3)
y_pred = kmeans.fit_predict(scaled_features)
y_pred  # so thses are the predicted categories of the data we provided to kmeans

df["SepalLengthCm"] = scaled_features[:, 0]
df["SepalWidthCm"] = scaled_features[:, 1]
df["PetalLengthCm"] = scaled_features[:, 2]
df["PetalWidthCm"] = scaled_features[:, 3]
df["Clusters"] = y_pred


scaled_features[:5]
df.head()  # Normalized dataset
df.describe()  # looking into the data for insights
# for a better understadin of the data lets take a look at the correlation between different features
sns.pairplot(df)


# Making Petal Clusters
pet_cluster1 = df[df['Clusters'] == 0].reset_index(drop=True)
pet_cluster1.head(3)
pet_cluster2 = df[df['Clusters'] == 1].reset_index(drop=True)
pet_cluster2.head(3)
pet_cluster3 = df[df['Clusters'] == 2].reset_index(drop=True)
pet_cluster3.head(3)

# Making Sepal Clusters
sep_cluster1 = df[df['Clusters'] == 0].reset_index(drop=True)
sep_cluster1.head(3)
sep_cluster2 = df[df['Clusters'] == 1].reset_index(drop=True)
sep_cluster2.head(3)
sep_cluster3 = df[df['Clusters'] == 2].reset_index(drop=True)
sep_cluster3.head(3)

# Plotting clusters


def plot_sep_cluster():
    plt.scatter(sep_cluster1.iloc[:, 2], sep_cluster1.iloc[:, 3], c='r',
                marker='o', edgecolors='black', label="Cluster-1")
    plt.scatter(sep_cluster2.iloc[:, 2], sep_cluster2.iloc[:, 3], c='b',
                marker='v', edgecolors='black', label="Cluster-2")
    plt.scatter(sep_cluster3.iloc[:, 2], sep_cluster3.iloc[:, 3], c='y',
                marker='s', edgecolors='black', label="Cluster-3")
    plt.legend

    centers = kmeans.cluster_centers_[:, -2:]  # cluster center for petals
    plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='X', s=200, label="Centroids")
    plt.figure(figsize=(12, 12))
    plt.show()


plot_sep_cluster()


def plot_pet_cluster():
    plt.scatter(pet_cluster1.iloc[:, 0], pet_cluster1.iloc[:, 1], c='r',
                marker='o', edgecolors='black', label="Cluster-1")
    plt.scatter(pet_cluster2.iloc[:, 0], pet_cluster2.iloc[:, 1], c='b',
                marker='v', edgecolors='black', label="Cluster-2")
    plt.scatter(pet_cluster3.iloc[:, 0], pet_cluster3.iloc[:, 1], c='y',
                marker='s', edgecolors='black', label="Cluster-3")
    plt.legend

    centers = kmeans.cluster_centers_[:, :-2]  # cluster center for petals
    centers
    plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='X', s=200, label="Centroids")
    plt.figure(figsize=(12, 12))
    plt.show()


plot_pet_cluster()


# -----------------------------------------------------------------------------
sep_cluster1
centers = kmeans.cluster_centers_[:, :-2]
centers
centers[:, 1]

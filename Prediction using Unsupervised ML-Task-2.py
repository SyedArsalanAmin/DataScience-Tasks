import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
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
    plt.title("Petal/Sepal anaylysis")


scatter_plot(df, 2, 3)  # visualizing petal data
scatter_plot(df, 0, 1)  # visualizing sepal data

df.describe()  # looking into the data for insights

# for a better understadin of the data lets take a look at the correlation between different features
sns.pairplot(df)

scatter_plot(df, 2, 3)  # visualizing petal data
scatter_plot(df, 0, 1)  # visualizing sepal data

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df)  # scaling dataframe
scaled_features.shape
scaled_features[:3]  # these are the normalized feature set between 0-1

# Using Elbow method to predict the no. of clusters


def elbow():

    cost = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit_predict(scaled_features)
        cost.append(kmeans.inertia_)

    plt.plot(np.arange(0, 10), cost, marker='o')
    plt.title("Elbow Method")
    plt.xlabel("No. of Clusters")
    plt.ylabel("Cost Function")


elbow()
# from thr above plot you can see clearly that there is not significant decrease
# in the cost so we should take 3 as the no. of cluster
# kmeans to preict the number of cluster
kmeans = KMeans(n_clusters=3)
y_pred = kmeans.fit_predict(scaled_features)
y_pred  # so thses are the predicted categories of the data we provided to kmeans

# Making normalized dataset
df["SepalLengthCm"] = scaled_features[:, 0]
df["SepalWidthCm"] = scaled_features[:, 1]
df["PetalLengthCm"] = scaled_features[:, 2]
df["PetalWidthCm"] = scaled_features[:, 3]
df["Clusters"] = y_pred
df.head()  # Normalized dataset

# Making Petal Clusters
pet_cluster1 = df[df['Clusters'] == 0].reset_index(drop=True)
pet_cluster1.head(3)
pet_cluster2 = df[df['Clusters'] == 1].reset_index(drop=True)
pet_cluster3 = df[df['Clusters'] == 2].reset_index(drop=True)

# Making Sepal Clusters
sep_cluster1 = df[df['Clusters'] == 0].reset_index(drop=True)
sep_cluster2 = df[df['Clusters'] == 1].reset_index(drop=True)
sep_cluster3 = df[df['Clusters'] == 2].reset_index(drop=True)


# Plotting clusters
def plot_sep_cluster():
    plt.figure(figsize=(15, 7))

    plt.scatter(sep_cluster1.iloc[:, 2], sep_cluster1.iloc[:, 3], c='r',
                marker='o', edgecolors='black', label="Cluster-1")
    plt.scatter(sep_cluster2.iloc[:, 2], sep_cluster2.iloc[:, 3], c='b',
                marker='v', edgecolors='black', label="Cluster-2")
    plt.scatter(sep_cluster3.iloc[:, 2], sep_cluster3.iloc[:, 3], c='y',
                marker='s', edgecolors='black', label="Cluster-3")

    centers = kmeans.cluster_centers_[:, -2:]  # cluster center for petals
    plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='X', s=200, label="Centroids")

    plt.xlabel('Length(cm)')
    plt.ylabel('Width(cm)')
    plt.legend()
    plt.title("Sepal Custer Anaylysis")
    plt.show()


plot_sep_cluster()


def plot_pet_cluster():
    plt.figure(figsize=(15, 7))

    plt.scatter(pet_cluster1.iloc[:, 0], pet_cluster1.iloc[:, 1], c='r',
                marker='o', edgecolors='black', label="Cluster-1")
    plt.scatter(pet_cluster2.iloc[:, 0], pet_cluster2.iloc[:, 1], c='b',
                marker='v', edgecolors='black', label="Cluster-2")
    plt.scatter(pet_cluster3.iloc[:, 0], pet_cluster3.iloc[:, 1], c='y',
                marker='s', edgecolors='black', label="Cluster-3")

    centers = kmeans.cluster_centers_[:, :-2]  # cluster center for petals
    centers
    plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='X', s=200, label="Centroids")

    plt.xlabel('Length(cm)')
    plt.ylabel('Width(cm)')
    plt.legend()
    plt.title("Petal Cluster Analysis")
    plt.show()


plot_pet_cluster()


# -----------------------------------------------------------------------------

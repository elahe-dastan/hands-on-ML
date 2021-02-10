from sklearn.cluster import KMeans
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

k = 5
kmeans = KMeans(n_clusters=k)
y_pred = kmeans.fit_predict(X)
print(kmeans.cluster_centers_)  # prints the centers of clusters
kmeans.transform(X)  # prints the distance from each point in X to each centroid

good_init = np.array([[-3, 3], [-1, 2]])
kmeans = KMeans(n_clusters=2, init=good_init, n_init=1)
print(kmeans.inertia_)  # inetria is mean squared distance between each instance and its closest centroid
print(kmeans.score(X))  # negative of inetria

minibatch_kmeans = MiniBatchKMeans(n_clusters=5)
minibatch_kmeans.fit(X)

print(silhouette_score(X, kmeans.labels_))  # varies from -1 to 1


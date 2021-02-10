from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA
import numpy as np

# get mnsit dataset
mnist = fetch_openml('mnist_784', version=1)
X = mnist["data"]
# plt.imshow(X[73].reshape(28, 28))
# plt.show()

# pca = PCA(n_components=154)
# X_reduced = pca.fit_transform(X)
# X_recovered = pca.inverse_transform(X_reduced)
# plt.imshow(X_recovered[73].reshape(28, 28))
# plt.show()

n_batches = 100
inc_pca = IncrementalPCA(n_components=154)
for X_batch in np.array_split(X, n_batches):
    inc_pca.partial_fit(X_batch)

X_reduced = inc_pca.transform(X)

# I can use memmap alternatively

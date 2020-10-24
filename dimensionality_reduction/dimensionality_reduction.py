import numpy as np
from sklearn.decomposition import PCA

X = np.array([[3, 2, 2],
              [2, 3, -2]])

# X_centered = X - X.mean(axis=0)
# U, s, Vt = np.linalg.svd(X_centered)
# c1 = Vt.T[:, 0]
# c2 = Vt.T[:, 1]
#
# W2 = Vt.T[:, :2]
# X2D = X_centered.dot(W2)

pca = PCA(n_components=2)
X2D = pca.fit_transform(X)
print(pca.components_)

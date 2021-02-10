from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np
from numpy.random import randint
from sklearn.metrics import mean_squared_error
from sklearn.manifold import LocallyLinearEmbedding
import matplotlib.pyplot as plt

X, _ = make_swiss_roll(n_samples=100)  # returns 100 samples with 3 features
# y = randint(0, 2, 100)  # specifies the class of the 100 samples
#
# clf = Pipeline([
#     ("kpca", KernelPCA(n_components=2)),
#     ("log_reg", LogisticRegression())
# ])
#
# param_grid = [{
#     "kpca__gamma": np.linspace(0.03, 0.05, 10),
#     "kpca__kernel": ["rbf", "sigmoid"]
# }]
#
# grid_search = GridSearchCV(clf, param_grid)
# grid_search.fit(X, y)
#
# print(grid_search.best_params_)

# reconstruction error (preimage)
# rbf_pca = KernelPCA(n_components=2, kernel="rbf", gamma=0.03, fit_inverse_transform=True)
# X_reduced = rbf_pca.fit_transform(X)
# X_preimage = rbf_pca.inverse_transform(X_reduced)
#
# print(mean_squared_error(X, X_preimage))

lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10)
X_reduced = lle.fit_transform(X)

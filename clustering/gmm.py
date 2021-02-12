from sklearn.mixture import GaussianMixture
import numpy as np

gm = GaussianMixture(n_components=3, n_init=10)
gm.fit(X)
print(gm.weights_)
print(gm.means_)
print(gm.covariances_)
print(gm.converged_)
print(gm.n_iter_)

print(gm.predict(X))
print(gm.predict_proba(X))

# This is a generative model
X_new, y_new = gm.sample(6)
print(X_new)
print(y_new)

# Anomaly detection
densities = gm.score_samples(X)
density_threshold = np.percentile(densities, 4)
anomalies = X[densities < density_threshold]

print(gm.aic(X))
print(gm.bic(X))

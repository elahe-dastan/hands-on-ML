from sklearn.mixture import BayesianGaussianMixture
import numpy as np

bgm = BayesianGaussianMixture(n_components=10, n_init=10, random_state=42)
bgm.fit(X)
np.round(bgm.weights_, 2)

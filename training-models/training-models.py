import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn import datasets


def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.show()


# X = 2 * np.random.rand(100, 1)
# y = 4 + 3 * X + np.random.randn(100, 1)

# plt.scatter(X, y)
# plt.show()

# X_b = np.c_[np.ones((100, 1)), X]
# theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
# theta_best = np.linalg.pinv(X_b).dot(y)
# print(theta_best)

# X_new = np.array([[0], [2]])
# X_new_b = np.c_[np.ones((2, 1)), X_new]
# y_predict = X_new_b.dot(theta_best)

# plt.plot(X_new, y_predict, "r-")
# plt.plot(X, y, "b.")
# plt.axis([0, 2, 0, 15])
# plt.show()

# lin_reg = LinearRegression()
# lin_reg.fit(X, y)
# print(lin_reg.intercept_)
# print(lin_reg.coef_)
# print(lin_reg.predict(X_new))

# theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=1e-6)
# print(theta_best_svd)

# eta = 0.1
# n_iterations = 1000
# m = 100
#
# theta = np.random.randn(2, 1)
#
# for iteration in range(n_iterations):
#     gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
#     theta = theta - eta * gradients
#
# print(theta)

# n_epochs = 50
# t0, t1 = 5, 50
#
#
# def learning_schedule(t):
#     return t0 / (t + t1)
#
#
# theta = np.random.randn(2, 1)
#
# for epoch in range(n_epochs):
#     for i in range(m):
#         random_index = np.random.randint(m)
#         xi = X_b[random_index:random_index + 1]
#         yi = y[random_index:random_index + 1]
#         gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
#         eta = learning_schedule(epoch * m + i)
#         theta = theta - eta * gradients
#
# print(theta)

# sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1)
# sgd_reg.fit(X, y.ravel())
# print(sgd_reg.intercept_)
# print(sgd_reg.coef_)

# m = 100
# X = 6 * np.random.rand(m, 1) - 3
# y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

# poly_features = PolynomialFeatures(degree=2, include_bias=False)
# X_poly = poly_features.fit_transform(X)
#
# lin_reg = LinearRegression()
# lin_reg.fit(X_poly, y)
# print(lin_reg.intercept_)
# print(lin_reg.coef_)
#
# y_new = lin_reg.predict(X_poly)
#
# plt.plot(X, y, "b.")
# plt.plot(X, y_new, "r.")
# plt.show()

# lin_reg = LinearRegression()
# plot_learning_curves(lin_reg, X, y)
#
# polynomial_regression = Pipeline([
#     ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
#     ("lin_reg", LinearRegression())
# ])
#
# plot_learning_curves(polynomial_regression, X, y)

# ridge_reg = Ridge(alpha=1, solver="cholesky")
# ridge_reg.fit(X, y)
# print(ridge_reg.predict([[1.5]]))
#
# sgd_reg = SGDRegressor(penalty="l2")
# sgd_reg.fit(X, y.ravel())
# print(sgd_reg.predict([[1.5]]))
#
# lasso_reg = Lasso(alpha=0.1)
# lasso_reg.fit(X, y)
# print(lasso_reg.predict([[1.5]]))
#
# elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
# elastic_net.fit(X, y)
# print(elastic_net.predict([[1.5]]))
#
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
#
# poly_scaler = Pipeline([
#     ("poly_features", PolynomialFeatures(degree=90, include_bias=False)),
#     ("std_scaler", StandardScaler())
# ])
#
# X_train_poly_scaled = poly_scaler.fit_transform(X_train)
# X_val_poly_scaled = poly_scaler.transform(X_val)
#
# sgd_reg = SGDRegressor(max_iter=1, tol=-np.infty, warm_start=True, penalty=None, learning_rate="constant", eta0=0.0005)
# minimum_val_error = float("inf")
# best_epoch = None
# best_model = None
# for epoch in range(1000):
#     sgd_reg.fit(X_train_poly_scaled, y_train.ravel())
#     y_val_predict = sgd_reg.predict(X_val_poly_scaled)
#     val_error = mean_squared_error(y_val, y_val_predict)
#     if val_error < minimum_val_error:
#         minimum_val_error = val_error
#         best_epoch = epoch
#         best_model = clone(sgd_reg)
#
# print(sgd_reg.predict(poly_scaler.transform([[1.5]])))

iris = datasets.load_iris()
print(list(iris.keys()))
# print(iris["data"])
# print(iris["target"])
# print(iris["target_names"])
# print(iris["DESCR"])
# print(iris["feature_names"])
# print(iris["filename"])
X = iris["data"][:, 3:]
y = (iris["target"] == 2).astype(np.int)

log_reg = LogisticRegression()
log_reg.fit(X, y)

X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)
# print(log_reg.classes_)
# plt.plot(X_new, y_proba[:, 1], "g-", label="Iris-Virginica")
# plt.plot(X_new, y_proba[:, 0], "b--", label="Not Iris-Virginica")
# plt.show()
print(log_reg.predict([[1.7], [1.5]]))





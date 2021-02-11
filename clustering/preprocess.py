from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV

X_digits, y_digits = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits)

# without preprocess
# log_reg = LogisticRegression(random_state=42)
# log_reg.fit(X_train, y_train)
# print(log_reg.score(X_test, y_test))

# using kmeans for clustering as a preprocess
# pipeline = Pipeline([
#     ("kmeans", KMeans(n_clusters=50)),
#     ("log_reg", LogisticRegression())
# ])
# pipeline.fit(X_train, y_train)
# print(pipeline.score(X_test, y_test))

pipeline = Pipeline([
    ("kmeans", KMeans()),
    ("log_reg", LogisticRegression())
])

param_grid = dict(kmeans__n_clusters=range(2, 100))
grid_clf = GridSearchCV(pipeline, param_grid, cv=3, verbose=2)
grid_clf.fit(X_train, y_train)
print(grid_clf.best_params_)
print(grid_clf.score(X_test, y_test))

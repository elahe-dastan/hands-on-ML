from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree import export_graphviz
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score

# iris = load_iris()
# X = iris.data[:, 2:]
# y = iris.target
#
# tree_clf = DecisionTreeClassifier(max_depth=2)
# tree_clf.fit(X, y)

# export_graphviz(
#     tree_clf,
#     out_file="iris_tree.dot",
#     feature_names=iris.feature_names[2:],
#     class_names=iris.target_names,
#     rounded=True,
#     filled=True
# )

# print(tree_clf.predict_proba([[5, 1.5]]))
# print(tree_clf.predict([[5, 1.5]]))

# m = 100
# X = 6 * (np.random.rand(m, 1) - 0.5)
# y = X**2 + np.random.randn(m, 1)
#
# tree_reg = DecisionTreeRegressor(max_depth=5)
# tree_reg.fit(X, y)
#
# print(tree_reg.predict([[2]]))
#
# export_graphviz(
#     tree_reg,
#     out_file="reg_tree.dot",
#     feature_names="X",
#     class_names="y",
#     rounded=True
# )
#
# plt.scatter(X, y)
# plt.show()

# X, y = make_moons(n_samples=100, shuffle=True, noise=1, random_state=None)
# X_train, X_test, y_train, y_test = X[:80], X[80:], y[:80], y[80:]
#
# log_clf = LogisticRegression()
# rnd_clf = RandomForestClassifier()
# svm_clf = SVC(probability=True)
#
# voting_clf = VotingClassifier(
#     estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
#     voting='soft'
# )
#
# for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
#     clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_test)
#     print(clf.__class__.__name__, accuracy_score(y_test, y_pred))






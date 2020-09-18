from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree import export_graphviz
import numpy as np
import matplotlib.pyplot as plt

# iris = load_iris()
# X = iris.data[:, 2:]
# y = iris.target
#
# tree_clf = DecisionTreeClassifier(max_depth=2)
# tree_clf.fit(X, y)
#
# export_graphviz(
#     tree_clf,
#     out_file="iris_tree.dot",
#     feature_names=iris.feature_names[2:],
#     class_names=iris.target_names,
#     rounded=True,
#     filled=True
# )
#
# print(tree_clf.predict_proba([[5, 1.5]]))
# print(tree_clf.predict([[5, 1.5]]))
#
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

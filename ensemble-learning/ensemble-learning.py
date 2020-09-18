from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import VotingClassifier, BaggingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# import xgboost

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
#
# bag_clf = BaggingClassifier(
#     DecisionTreeClassifier(), n_estimators=500,
#     max_samples=50, bootstrap=True, n_jobs=-1, oob_score=True
# )
#
# bag_clf.fit(X_train, y_train)
#
# print(bag_clf.oob_score_)
#
# y_pred = bag_clf.predict(X_test)
# print(accuracy_score(y_test, y_pred))
# print(bag_clf.oob_decision_function_)
#
# rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
# rnd_clf.fit(X_train, y_train)
# print(rnd_clf.predict(X_test))
#
# bag_clf = BaggingClassifier(DecisionTreeClassifier(splitter="random", max_leaf_nodes=16),
#                             n_estimators=500, max_samples=1.0, bootstrap=True, n_jobs=-1)
#
# iris = load_iris()
#
# rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
# rnd_clf.fit(iris["data"], iris["target"])
#
# for name, score in zip(iris["feature_names"], rnd_clf.feature_importances_):
#     print(name, score)
#
# ada_clf = AdaBoostClassifier(
#     DecisionTreeClassifier(max_depth=1), n_estimators=200,
#     algorithm="SAMME.R", learning_rate=0.5
# )
#
# ada_clf.fit(X_train, y_train)

X = 2 * np.random.rand(100, 1)
y = 4 + X**2 + np.random.randn(100, 1)

# tree_reg1 = DecisionTreeRegressor(max_depth=2)
# tree_reg1.fit(X, y)
#
# y2 = y - tree_reg1.predict(X).reshape(-1, 1)
# tree_reg2 = DecisionTreeRegressor(max_depth=2)
# tree_reg2.fit(X, y2)
#
# y3 = y2 - tree_reg2.predict(X).reshape(-1, 1)
# tree_reg3 = DecisionTreeRegressor(max_depth=2)
# tree_reg3.fit(X, y3)
#
# X_new = [[2]]
#
# y_pred = sum(tree.predict(X_new) for tree in (tree_reg1, tree_reg2, tree_reg3))
# print(y_pred)

# gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0)
# gbrt.fit(X, y)

X_train, X_val, y_train, y_val = train_test_split(X, y)

# gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120)
# gbrt.fit(X_train, y_train.ravel())
#
# errors = [mean_squared_error(y_val, y_pred) for y_pred in gbrt.staged_predict(X_val)]
#
# bst_n_estimators = np.argmin(errors)
#
# print(bst_n_estimators)
#
# gbrt_best = GradientBoostingRegressor(max_depth=2, n_estimators=bst_n_estimators)
# gbrt_best.fit(X_train, y_train.ravel())

# gbrt = GradientBoostingRegressor(max_depth=2, warm_start=True)
#
# min_val_error = float("inf")
# error_going_up = 0
# for n_estimators in range(1, 120):
#     gbrt.n_estimators = n_estimators
#     gbrt.fit(X_train, y_train)
#     y_pred = gbrt.predict(X_val)
#     val_error = mean_squared_error(y_val, y_pred)
#     if val_error < min_val_error:
#         min_val_error = val_error
#         error_going_up = 0
#     else:
#         error_going_up += 1
#         if error_going_up == 5:
#             break

# xgb_reg = xgboost.XGBRegressor()
# xgb_reg.fit(X_train, y_train)
# y_pred = xgb_reg.predict(X_val)
#
# xgb_reg.fit(X_train, y_train,
# eval_set=[(X_val, y_val)], early_stopping_rounds=2)
# y_pred = xgb_reg.predict(X_val)















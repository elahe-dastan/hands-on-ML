from sklearn.datasets import fetch_openml
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass

    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)


# The last precision and recall values are 1. and 0. respectively and do not
# have a corresponding threshold.
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')


mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]
y = y.astype(np.uint8)

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

print("got data")

sgd_clf = SGDClassifier(random_state=42)
# sgd_clf.fit(X_train, y_train_5)

# skfolds = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)
#
# for train_index, test_index in skfolds.split(X_train, y_train_5):
#     clone_clf = clone(sgd_clf)
#     X_train_folds = X_train[train_index]
#     y_train_folds = y_train_5[train_index]
#     X_test_fold = X_train[test_index]
#     y_test_fold = y_train_5[test_index]
#
#     clone_clf.fit(X_train_folds, y_train_folds)
#     y_pred = clone_clf.predict(X_test_fold)
#     n_correct = sum(y_pred == y_test_fold)
#     print(n_correct / len(y_pred))

# never_5_clf = Never5Classifier()
# print(cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy"))

# y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

# print(confusion_matrix(y_train_5, y_train_pred))
# print(precision_score(y_train_5, y_train_pred))
# print(recall_score(y_train_5, y_train_pred))
# print(f1_score(y_train_5, y_train_pred))


some_digit = X[0]
# print(some_digit.astype(np.float64))
# print(y_train[0])
# some_digit_image = some_digit.reshape(28, 28)
#
# plt.imshow(some_digit_image, cmap=mpl.cm.binary, interpolation="nearest")
# plt.axis("off")
# plt.show()

# y_score = sgd_clf.decision_function([some_digit])
# print(y_score)
# threshold = 0
# y_some_digit_pred = (y_score > threshold)
# print(y_some_digit_pred)
# threshold = 8000
# y_some_digit_pred = (y_score > threshold)
# print(y_some_digit_pred)

# y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
# precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
# print(precisions)
# print(recalls)
# print(thresholds)

# plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
# plt.show()

# threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]
#
# y_train_pred_90 = (y_scores >= threshold_90_precision)
# print(precision_score(y_train_5, y_train_pred_90))
# print(recall_score(y_train_5, y_train_pred_90))

# fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
#
# plot_roc_curve(fpr, tpr)
# plt.show()

# print(roc_auc_score(y_train_5, y_scores))

forest_clf = RandomForestClassifier(random_state=42)
# y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")
# y_scores_forest = y_probas_forest[:, 1] # score = proba of positive class
# fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)
#
# plt.plot(fpr, tpr, "b:", label="SGD")
# plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
# plt.show()

# print(roc_auc_score(y_train_5, y_scores_forest))

# sgd_clf.fit(X_train, y_train)
# print(sgd_clf.predict([some_digit]))
# some_digit_scores = sgd_clf.decision_function([some_digit])
# print(some_digit_scores)
#
# print(np.argmax(some_digit_scores))
# print(sgd_clf.classes_)

# ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
# ovo_clf.fit(X_train, y_train)
# print(ovo_clf.predict([some_digit]))
# print(len(ovo_clf.estimators_))

# forest_clf.fit(X_train, y_train)
# print(forest_clf.predict([some_digit]))
# print(forest_clf.predict_proba([some_digit]))

# print(cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy"))

# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# print("scaled")
# print(cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy"))

# y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
# conf_mx = confusion_matrix(y_train, y_train_pred)
# print(conf_mx)
# plt.matshow(conf_mx, cmap=plt.cm.gray)
# plt.show()

# row_sums = conf_mx.sum(axis=1, keepdims=True)
# norm_conf_mx = conf_mx / row_sums
# np.fill_diagonal(norm_conf_mx, 0)
# plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
# plt.show()

# cl_a, cl_b = 3, 5
# X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
# X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
# X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
# X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]
#
# plt.figure(figsize=(8, 8))
# plt.subplot(221)
# plt.imshow(X_aa[:25])
# plt.subplot(222)
# plt.imshow(X_ab[:25])
# plt.subplot(223)
# plt.imshow(X_ba[:25])
# plt.subplot(224)
# plt.imshow(X_bb[:25])
# plt.show()

# y_train_large = (y_train >= 7)
# y_train_odd = (y_train % 2 == 1)
# y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
# knn_clf.fit(X_train, y_multilabel)
# print(knn_clf.predict([some_digit]))
# y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
# f1_score(y_multilabel, y_train_knn_pred, average="macro")
# f1_score(y_multilabel, y_train_knn_pred, average="weighted")

noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test

knn_clf.fit(X_train_mod, y_train_mod)
plt.matshow(X_test_mod[20].reshape(28, 28), cmap=plt.cm.gray)
clean_digit = knn_clf.predict([X_test_mod[20]])
plt.matshow(clean_digit.reshape(28, 28), cmap=plt.cm.gray)
plt.show()


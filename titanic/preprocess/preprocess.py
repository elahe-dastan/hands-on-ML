import os
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
import csv
from sklearn.ensemble import RandomForestClassifier

TITANIC_PATH = os.path.join("/home/raha/py/src/hands-on-ML/titanic/datasets/")


def load_titanic_train_data(titanic_path=TITANIC_PATH):
    csv_path = os.path.join(titanic_path, "train.csv")
    return pd.read_csv(csv_path)


def load_titanic_test_data(titanic_path=TITANIC_PATH):
    csv_path = os.path.join(titanic_path, "test.csv")
    return pd.read_csv(csv_path)


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


data = load_titanic_train_data()
print(data.info())

test_data = load_titanic_test_data()

data_labels = data["Survived"].copy()

data = data.drop("Survived", axis=1).drop("Name", axis=1).drop("Cabin", axis=1).drop("Embarked", axis=1).drop("Ticket", axis=1)
test_data = test_data.drop("Name", axis=1).drop("Cabin", axis=1).drop("Embarked", axis=1).drop("Ticket", axis=1)
data_num = data.drop("Sex", axis=1)


# a = data["Cabin"][1]
# data = data["Cabin"].fillna(a, inplace=True)


num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, list(data_num)),
    ("cat_sex", OrdinalEncoder(), ["Sex"]),
])

data_prepared = full_pipeline.fit_transform(data)
print(data_prepared)

test_data_prepared = full_pipeline.transform(test_data)

# log_reg = LogisticRegression()
# log_reg.fit(data_prepared, data_labels)

forest_clf = RandomForestClassifier()
forest_clf.fit(data_prepared, data_labels)

# data_predictions = log_reg.predict(test_data_prepared)
# print(data_predictions)

data_predictions = forest_clf.predict(test_data_prepared)
print(data_predictions)

with open('/home/raha/py/src/hands-on-ML/titanic/datasets/result.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["PassengerId", "Survived"])
    for id, s in zip(test_data["PassengerId"], data_predictions):
        writer.writerow([id, s])

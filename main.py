
# Author:   Sebastian Law
# Date:     10-Mar-2016
# Revised:  10-Mar-2016

import features
import seaborn as sns
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import csv


def dump(passenger_id, y, file_name="result.csv"):
    with open(file_name, "w", newline='') as file:
        output = csv.writer(file)
        output.writerow(["PassengerId", "Survived"])
        output.writerows(zip(passenger_id, y))


# Load the train and test sets
data = features.get_data()
train = data.loc[data['Survived'].notnull()]
train_features = train.values[:, 2:]
train_labels = train.values[:, 1]
test = data.loc[data['Survived'].isnull()]
test_features = test.values[:, 2:]
p_id = test['PassengerId']

# Set up classifier (Grid search Mean score: 0.8451 (std: 0.0380))
forest = RandomForestClassifier(n_estimators=3000,
                                max_depth=11,
                                criterion='entropy',
                                min_samples_split=2,
                                max_features=7,
                                random_state=101)

# Cross validation
skf = cross_validation.StratifiedKFold(train_labels, n_folds=5)
cv_predict = cross_validation.cross_val_predict(forest, train_features, train_labels, cv=skf)
print("Accuracy:", metrics.accuracy_score(train_labels, cv_predict))
print(metrics.classification_report(train_labels, cv_predict))

# Train and predict
forest = forest.fit(X=train_features, y=train_labels)
test_predictions = forest.predict(test_features).astype(int)
dump(p_id, test_predictions)



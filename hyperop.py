
# Author:   Sebastian Law
# Date:     22-Mar-2016
# Revised:  22-Mar-2016

import features
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from operator import itemgetter

np.seterr(invalid='ignore')  # stops error:: RuntimeWarning: invalid value encountered in greater


def report(grid_scores, n_top=6):
    params = None
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Rank: {0}".format(i + 1))
        print("Mean score: {0:.4f} (std: {1:.4f})".format(
              score.mean_validation_score, np.std(score.cv_validation_scores)))
        print("Parameters:", score.parameters)
        print("")
        if params is None:
            params = score.parameters
    return params

data = features.get_data()
train = data.loc[data['Survived'].notnull()]
X = train.values[:, 2:]
y = train.values[:, 1]

sqrtfeat = np.sqrt(X.shape[1]).astype(int)

grid_test = {"n_estimators"      : [1000, 2000, 3000, 4000, 5000],
             "criterion"         : ["gini", "entropy"],
             "max_features"      : [sqrtfeat, sqrtfeat+1, sqrtfeat+2, sqrtfeat+3],
             "max_depth"         : [5, 7, 9, 11, 13],
             "min_samples_split" : [2, 4, 6, 8, 10]}

forest = RandomForestClassifier(oob_score=True)

grid_search = GridSearchCV(forest, grid_test, n_jobs=-1, cv=10)
grid_search.fit(X, y)
report(grid_search.grid_scores_)


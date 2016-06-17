
# Author:   Sebastian Law
# Date:     30-Mar-2016
# Revised:  30-Mar-2016

import features
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve

data = features.get_data()
train = data.loc[data['Survived'].notnull()]
X = train.values[:, 2:]
y = train.values[:, 1]

# forest = RandomForestClassifier(n_estimators=1000,
#                                 max_depth=8,
#                                 criterion='entropy',
#                                 min_samples_split=5,
#                                 max_features=6)
forest = RandomForestClassifier(n_estimators=1000,
                                max_depth=9,
                                criterion='entropy',
                                min_samples_split=10,
                                max_features=6)

train_sizes, train_scores, test_scores = learning_curve(
    forest, X, y, cv=10, n_jobs=3, train_sizes=np.linspace(.1, 1., 20), verbose=0)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure()
plt.xlabel("training examples")
plt.ylabel("score")
plt.ylim((0.7, 1.0))
plt.gca().invert_yaxis()
plt.grid()
plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label="train")
plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label="test")
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,alpha=0.1, color="b")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,alpha=0.1, color="r")
plt.draw()
plt.legend(loc="best")

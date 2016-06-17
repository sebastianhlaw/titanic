
# Author:   Sebastian Law
# Date:     10-Mar-2016
# Revised:  22-Mar-2016

import os
import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

pd.options.mode.chained_assignment = None  # default='warn'


def get_data(fare_bins=6, age_bins=6):
    # load the data
    data_path = os.path.join(os.path.expanduser('~'), 'Google Drive', 'Datasets', 'Kaggle',
                             'Titanic Machine Learning from Disaster')
    train = pd.read_csv(os.path.join(data_path, 'train.csv'), header=0)
    test = pd.read_csv(os.path.join(data_path, 'test.csv'), header=0)
    data = pd.concat([train, test], axis=0)
    cols = list(data)
    cols.insert(0, cols.pop(cols.index('PassengerId')))
    cols.insert(1, cols.pop(cols.index('Survived')))
    data = data.ix[:, cols]
    
    # replace genders in both data sets with numeric values
    data['Sex'] = data['Sex'].map({'female': 0, 'male': 1}).astype(np.int8)
    
    # replace ports in both data sets with numeric values. Null values converted to unique value
    train.Embarked.loc[train.Embarked.isnull()] = train.Embarked.mode()
    data = pd.concat([data, pd.get_dummies(data['Embarked']).rename(columns=lambda x: 'Embarked_' + str(x))], axis=1)
    
    # Pull out the deck level from the cabin ids * could also use pd.factorize
    data['Cabin'][data['Cabin'].isnull()] = 'U0'
    data['Deck'] = data['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    data['DeckID'] = pd.factorize(data['Deck'], sort=True)[0]
    
    # Pad missing fares with class median
    median_fare = np.zeros(3)
    for f in range(0, 3):  # could add clairvoyance
        median_fare[f] = data[data['Pclass'] == f + 1]['Fare'].dropna().median()
    for f in range(0, 3):  # all fares present in training set
        data.loc[(data.Fare.isnull()) & (data.Pclass == f + 1), 'Fare'] = median_fare[f]
    # Stick fares into quantiles (cheating a bit, using test bin data too)
    # n_bins = 6
    data['FareBin'], bins = pd.qcut(data['Fare'], fare_bins, labels=[x for x in range(fare_bins)], retbins=True)
    # data['FareBin'] = pd.cut(data['Fare'], bins, labels=[x for x in range(n_bins)], include_lowest=True)
    data['FareBinID'] = pd.factorize(data['FareBin'], sort=True)[0]
    data = pd.concat([data, pd.get_dummies(data['FareBinID']).astype(np.int8).rename(columns=lambda x: 'FareBin_'+str(x))],
                     axis=1)
    
    # Group titles together and create binary columns
    data['Title'] = data['Name'].map(lambda x: re.compile(", (.*?)\.").findall(x)[0]).str.upper()
    data['Title'][data.Title.isin(['MS', 'MLLE', 'MME'])] = 'MISS'
    data['Title'][data.Title.isin(['CAPT', 'COL', 'MAJOR', 'DON', 'JONKHEER'])] = 'SIR'
    data['Title'][data.Title.isin(['DONA', 'THE COUNTESS'])] = 'LADY'
    data = pd.concat([data, pd.get_dummies(data['Title']).astype(np.int8).rename(columns=lambda x: 'Title_'+str(x))],
                     axis=1)
    
    # linear regresssion to fit the ages.
    data_known_age = data.loc[data['Age'].notnull()].select_dtypes(include=[np.number])
    data_unknown_age = data.loc[data['Age'].isnull()].select_dtypes(include=[np.number])
    X_train = data_known_age.values[:, 3::]
    y_train = data_known_age.values[:, 2]
    L = LinearRegression().fit(X_train, y_train)
    X_fit = data_unknown_age.values[:, 3::]
    y_fit = L.predict(X_fit)
    data.loc[data['Age'].isnull(), 'Age'] = y_fit
    data['AgeBin'], bins = pd.qcut(data['Age'], age_bins, labels=[x for x in range(age_bins)], retbins=True)
    data['AgeBinID'] = pd.factorize(data['AgeBin'], sort=True)[0]
    data = pd.concat([data, pd.get_dummies(data['AgeBinID']).astype(np.int8).rename(columns=lambda x: 'AgeBin_'+str(x))],
                     axis=1)

    # drop remaining useless data
    data = data.select_dtypes(include=[np.number])
    return data


def feature_engineering(data, correlation_cutoff=0.9):
    # scale a bunch of the numeric data
    scaler = preprocessing.StandardScaler()
    scaling_list = ['Age', 'Fare', 'Pclass', 'Parch', 'SibSp', 'DeckID', 'AgeBinID', 'FareBinID']
    for i in scaling_list:
        data[i+'X'] = scaler.fit_transform(data[i].astype(float))

    # generate non-linear features
    scaled_list = [i+'X' for i in scaling_list]
    data_numeric = data.loc[:, scaled_list]
    for i in range(data_numeric.columns.size-1):
        for j in range(data_numeric.columns.size-1):
            ci = data_numeric.columns.values[i]
            cj = data_numeric.columns.values[j]
            # products
            if i <= j:
                cx = ci+'*'+cj
                data = pd.concat([data, pd.Series(data_numeric.iloc[:, i] * data_numeric.iloc[:, j], name=cx)], axis=1)
            # sums
            if i < j:
                cx = ci+'+'+cj
                data = pd.concat([data, pd.Series(data_numeric.iloc[:, i] + data_numeric.iloc[:, j], name=cx)], axis=1)
            # division and subtraction
            if i != j:
                cx = ci+'-'+cj
                data = pd.concat([data, pd.Series(data_numeric.iloc[:, i] - data_numeric.iloc[:, j], name=cx)], axis=1)
                cx = ci+'/'+cj
                data = pd.concat([data, pd.Series(data_numeric.iloc[:, i] / data_numeric.iloc[:, j], name=cx)], axis=1)

    # drop highly correlated data
    correlations = data.drop(['PassengerId', 'Survived'], axis=1).corr(method='spearman')
    mask = np.ones(correlations.columns.size) - np.eye(correlations.columns.size)
    correlations = mask * correlations
    drops = []
    for i in correlations.columns.values:
        if np.in1d([i], drops):
            continue
        c = correlations[abs(correlations[i]) > correlation_cutoff].index
        drops = np.union1d(drops, c)
    data.drop(drops, axis=1, inplace=True)
    print("Dropped", drops.shape[0], "features, rho >", correlation_cutoff, "leaving", data.columns.size, "features")

    # train = data.loc[data['Survived'].notnull()]
    # features = train.columns.values[2:]
    # X = train.values[:, 2:]
    # y = train.values[:, 1]
    # forest = RandomForestClassifier(oob_score=True, n_estimators=10000)
    # forest.fit(X, y)
    # feature_importance = forest.feature_importances_
    return data


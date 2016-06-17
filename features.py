
# Author:   Sebastian Law
# Date:     10-Mar-2016
# Revised:  31-Mar-2016

import os
import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None  # default='warn'


def get_data():
    data = get_basic_data()
    data = generate_bins(data, "Age", n_bins=6, binary_data=False, drop_original=True)
    data = generate_bins(data, "Fare", n_bins=6, binary_data=False, drop_original=True)
    data = data.drop('SurnameID', axis=1)
    return data

def get_basic_data():
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
    data = pd.concat(
        [data, pd.get_dummies(data['Embarked']).rename(columns=lambda x: 'Embarked_' + str(x)).astype(np.int8)], axis=1)
    
    # Pull out the deck level and Location from the cabin
    data['Cabin'][data['Cabin'].isnull()] = 'U0'
    data['Deck'] = data['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    data['DeckID'] = pd.factorize(data['Deck'], sort=True)[0]
    f = lambda x: re.compile("(\d+)").search(x).group() if re.compile("(\d+)").search(x) else 0
    data['Location'] = data['Cabin'].map(f).astype(int)
    
    # Pad missing fares with class median
    median_fare = np.zeros(3)
    for f in range(0, 3):  # could add clairvoyance
        median_fare[f] = data[data['Pclass'] == f + 1]['Fare'].dropna().median()
    for f in range(0, 3):  # all fares present in training set
        data.loc[(data.Fare.isnull()) & (data.Pclass == f + 1), 'Fare'] = median_fare[f]

    # Group titles together and create binary columns
    data['Title'] = data['Name'].map(lambda x: re.compile(", (.*?)\.").findall(x)[0]).str.upper()
    data['Title'][data.Title.isin(['MS', 'MLLE', 'MME'])] = 'MISS'
    data['Title'][data.Title.isin(['CAPT', 'COL', 'MAJOR', 'DON', 'JONKHEER'])] = 'SIR'
    data['Title'][data.Title.isin(['DONA', 'THE COUNTESS'])] = 'LADY'
    titles = ["Title_"+x for x in data['Title'].unique()]
    data = pd.concat([data, pd.get_dummies(data['Title']).astype(np.int8).rename(columns=lambda x: 'Title_'+str(x))],
                     axis=1)
    data['Surname'] = data['Name'].map(lambda x: re.compile("^([^,]*)").search(x).group())
    data['SurnameID'] = pd.factorize(data['Surname'])[0]
    data['SurnameCount'] = data['Surname'].map(data['Surname'].value_counts())
    
    # linear regresssion to fit the ages.
    # data_age = data.drop(['SurnameID', 'SurnameCount'], axis=1)
    data_age = data[['Age', 'Parch', 'SibSp']+titles]
    data_known_age = data_age.loc[data_age['Age'].notnull()]  # .select_dtypes(include=[np.number])
    data_unknown_age = data_age.loc[data_age['Age'].isnull()]  # .select_dtypes(include=[np.number])
    X_train = data_known_age.values[:, 1::]
    y_train = data_known_age.values[:, 0]
    L = LinearRegression().fit(X_train, y_train)
    X_fit = data_unknown_age.values[:, 1::]
    y_fit = L.predict(X_fit)
    data.loc[data_age['Age'].isnull(), 'Age'] = y_fit

    # drop remaining useless data
    data = data.select_dtypes(include=[np.number])
    return data


def generate_bins(data, category, n_bins=6, binary_data=False, drop_original=True):
    # bin float data
    category_bin = category+"Bin"
    category_id = category_bin+"ID"
    data[category_bin], bins = pd.qcut(data[category], n_bins, labels=[x for x in range(n_bins)], retbins=True)
    data[category_id] = pd.factorize(data[category_bin], sort=True)[0]
    data = data.drop(category_bin, axis=1)
    if binary_data:
        data = pd.concat([data, pd.get_dummies(data[category_id]).astype(np.int8)
                         .rename(columns=lambda x: category_bin+"_"+str(x))], axis=1)
        data = data.drop(category_id, axis=1)
    if drop_original:
        data = data.drop(category, axis=1)
    return data


def feature_generation(data, correlation_cutoff=0.9):
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
    return data

def feature_importance(data, importance_threshold = 0.0):
    # plot relative feature importances
    data = get_data()
    train = data.loc[data['Survived'].notnull()]
    features = train.columns.values[2:]
    X = train.values[:, 2:]
    y = train.values[:, 1]
    # forest = RandomForestClassifier(n_estimators=1000,
    #                                 max_depth=8,
    #                                 criterion='entropy',
    #                                 min_samples_split=5,
    #                                 max_features=6,
    #                                 oob_score=True)
    forest = RandomForestClassifier(n_estimators=1000, oob_score=True)
    forest.fit(X, y)
    print("oob score", forest.oob_score_)
    feature_weights = forest.feature_importances_
    feature_weights = feature_weights / feature_weights.max()
    important_i = np.where(feature_weights > importance_threshold)[0]
    important_features = features[important_i]
    sorted_i = np.argsort(feature_weights[important_i])[::-1]
    sorted_weights = feature_weights[important_i][sorted_i[::-1]]
    sorted_features = important_features[sorted_i[::-1]]
    # plotting
    pos = np.arange(sorted_i.shape[0]) + 0.5
    plt.barh(pos, sorted_weights, align='center')
    plt.yticks(pos, sorted_features)
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.draw()
    plt.show()
    return sorted_features[::-1]


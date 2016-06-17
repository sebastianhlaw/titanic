
# Author:   Sebastian Law
# Date:     10-Mar-2016
# Revised:  16-Mar-2016

import os
import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LinearRegression

pd.options.mode.chained_assignment = None  # default='warn'


def get_data():
    """Load the Titanic data set and clean up features

    Returns
    -------
    train : pandas dataframe, shape [n samples, 1 label + k features].
    test : pandas dataframe, shape [m samples, k features].

    Improvements
    ------------
    - Multiple cabins sometimes listed under one ticket, which could be shared among others.
    - Number in cabin info linked to location within ship
    """

    # Cheat a bit or not? (look at test data
    clairvoyant = True

    # load the data
    data_path = os.path.join(os.path.expanduser('~'), 'Google Drive', 'Datasets', 'Kaggle',
                             'Titanic Machine Learning from Disaster')
    train = pd.read_csv(os.path.join(data_path, 'train.csv'), header=0)
    test = pd.read_csv(os.path.join(data_path, 'test.csv'), header=0)
    data = pd.concat([train, test], axis=0)

    # drop wholly useless data (not strictly so, as seems to contain class etc., group nannies with families etc.)
    train = train.drop(['Ticket'], axis=1)
    test = test.drop(['Ticket'], axis=1)

    # replace genders in both data sets with numeric values
    train['Sex'] = train['Sex'].map({'female': 0, 'male': 1}).astype(int)
    test['Sex'] = test['Sex'].map({'female': 0, 'male': 1}).astype(int)

    # replace ports in both data sets with numeric values. Null values converted to unique value
    train.Embarked.loc[train.Embarked.isnull()] = train.Embarked.mode()
    train = pd.concat([train, pd.get_dummies(train['Embarked']).rename(columns=lambda x: 'Embarked_' + str(x))], axis=1)
    test = pd.concat([test, pd.get_dummies(test['Embarked']).rename(columns=lambda x: 'Embarked_' + str(x))], axis=1)

    # Pull out the deck level from the cabin ids * could also use pd.factorize
    train['Deck'] = train['Cabin'].str[:1].str.upper()
    test['Deck'] = test['Cabin'].str[:1].str.upper()
    decks = list(enumerate(train['Deck'].unique()))
    decks_dict = {deck: i for i, deck in decks}
    train['Deck'] = train['Deck'].map(lambda x: decks_dict[x]).astype(int)
    test['Deck'] = test['Deck'].map(lambda x: decks_dict[x]).astype(int)

    # Pad missing fares with class median
    median_fare = np.zeros(3)
    for f in range(0, 3):  # could add clairvoyance
        median_fare[f] = train[train['Pclass'] == f + 1]['Fare'].dropna().median()
    for f in range(0, 3):  # all fares present in training set
        test.loc[(test.Fare.isnull()) & (test.Pclass == f + 1), 'Fare'] = median_fare[f]

    # Stick fares into quantiles (cheating a bit, using test bin data too)
    n_bins = 8
    if clairvoyant:
        all_fares = pd.concat([train['Fare'], test['Fare']], axis=0, ignore_index=True)
        temp, bins = pd.qcut(all_fares, n_bins, labels=[x for x in range(n_bins)], retbins=True)
        train['FareBin'] = pd.cut(train['Fare'], bins, labels=[x for x in range(n_bins)], include_lowest=True)
    else:
        train['FareBin'], bins = pd.qcut(train['Fare'], n_bins, labels=[x for x in range(n_bins)], retbins=True)
    test['FareBin'] = pd.cut(test['Fare'], bins, labels=[x for x in range(n_bins)], include_lowest=True)
    train['FareBinId'] = pd.factorize(train['FareBin'], sort=True)[0]
    test['FareBinId'] = pd.factorize(test['FareBin'], sort=True)[0]

    # Group titles together and create binary columns
    train['Title'] = train['Name'].map(lambda x: re.compile(", (.*?)\.").findall(x)[0]).str.upper()
    test['Title'] = test['Name'].map(lambda x: re.compile(", (.*?)\.").findall(x)[0]).str.upper()
    train['Title'][train.Title.isin(['MS', 'MLLE', 'MME'])] = 'MISS'
    train['Title'][train.Title.isin(['CAPT', 'COL', 'MAJOR', 'DON', 'JONKHEER'])] = 'SIR'
    train['Title'][train.Title.isin(['DONA', 'THE COUNTESS'])] = 'LADY'
    test['Title'][test.Title.isin(['MS', 'MLLE', 'MME'])] = 'MISS'
    test['Title'][test.Title.isin(['CAPT', 'COL', 'MAJOR', 'DON', 'JONKHEER'])] = 'SIR'
    test['Title'][test.Title.isin(['DONA', 'THE COUNTESS'])] = 'LADY'
    train = pd.concat([train, pd.get_dummies(train['Title']).rename(columns=lambda x: 'Title_' + str(x))], axis=1)
    test = pd.concat([test, pd.get_dummies(test['Title']).rename(columns=lambda x: 'Title_' + str(x))], axis=1)

    # simple ages mapping based on title and mean age for that title
    # titles = list(train['Title'].unique())
    # ages = [train[(train['Title'] == x)]['Age'].mean() for x in titles]
    # ages_dict = dict(zip(titles, ages))
    # train['SimpleAge'] = train.loc[train['Age'].notnull(), 'Age']
    # test['SimpleAge'] = test.loc[test['Age'].notnull(), 'Age']
    # train.loc[train['Age'].isnull(), 'SimpleAge'] = train.loc[train['Age'].isnull(), 'Title'].map(lambda x: ages_dict[x])
    # test.loc[test['Age'].isnull(), 'SimpleAge'] = test.loc[test['Age'].isnull(), 'Title'].map(lambda x: ages_dict[x])

    # drop remaining useless data
    train = train.drop(['Embarked', 'Name', 'Cabin', 'Title', 'Fare', 'FareBin'], axis=1)
    test = test.drop(['Embarked', 'Name', 'Cabin', 'Title', 'Fare', 'FareBin'], axis=1)

    # rearrange columns
    cols = list(train)
    cols.insert(2, cols.pop(cols.index('Age')))
    train = train.ix[:, cols]
    cols = list(test)
    cols.insert(1, cols.pop(cols.index('Age')))
    test = test.ix[:, cols]
    # linear fit for ages
    known_train = train.loc[train['Age'].notnull()]
    unknown_train = train.loc[train['Age'].isnull()]
    X_train = known_train.values[:, 3::]
    y_train = known_train.values[:, 2]
    L = LinearRegression().fit(X_train, y_train)
    # fill in missing training set ages
    X0_train = unknown_train.values[:, 3::]
    y0_train = L.predict(X0_train)
    train.loc[train['Age'].isnull(), 'Age'] = y0_train
    # fill in missing testing set ages
    unknown_test = test.loc[test['Age'].isnull()]
    X0_test = unknown_test.values[:, 2::]
    y0_test = L.predict(X0_test)
    test.loc[test['Age'].isnull(), 'Age'] = y0_test

    return train, test

if __name__ == "__main__":
    train_df, test_df = get_data()
    # feature generation







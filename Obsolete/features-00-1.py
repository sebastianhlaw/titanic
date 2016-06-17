
# Author:   Sebastian Law
# Date:     10-Mar-2016

import os
import pandas as pd
import numpy as np


pd.options.mode.chained_assignment = None  # default='warn'


def get_data():
    """Load the Titanic data set and clean up features

    - Sex mapped from 'male', 'female' to 1, 0.
    - Embarkation port mapped from 'S', 'Q', 'C' to integers.
    - Cabin letter (i.e. Deck) mapped to integers (this is sparse data).
    - Null Ages padded using average aged by combination of Title & Sex.
    - Null Fares padded with the median of the class

    Returns
    -------
    train : pandas dataframe, shape [n samples, 1 label + k features].
    test : pandas dataframe, shape [m samples, k features].

    Improvements
    ------------
    - Multiple cabins sometimes listed under one ticket, which could be shared among others.
    - Number in cabin info linked to location within ship
    """

    # Load the data #

    data_path = os.path.join(os.path.expanduser('~'), 'Google Drive', 'Datasets', 'Kaggle',
                             'Titanic Machine Learning from Disaster')
    train = pd.read_csv(os.path.join(data_path, 'train.csv'), header=0)
    test = pd.read_csv(os.path.join(data_path, 'test.csv'), header=0)
    print("Training samples:", len(train), "fraction:", '{:f}'.format(len(train)/(len(train)+len(test))))
    print("Testing samples:", len(test), "fraction:", '{:f}'.format(len(test)/(len(train)+len(test))))

    # Pre-processing #

    # drop wholly useless data
    train = train.drop(['Ticket'], axis=1)
    test = test.drop(['Ticket'], axis=1)

    # replace genders in both data sets with numeric values
    train['Sex'] = train['Sex'].map({'female': 0, 'male': 1}).astype(int)
    test['Sex'] = test['Sex'].map({'female': 0, 'male': 1}).astype(int)

    # replace ports in both data sets with numeric values. Null values converted to unique value
    if len(train.Embarked[train.Embarked.isnull()]) > 0:
        train.Embarked[train.Embarked.isnull()] = "X"  # train.Embarked.dropna().mode().values
    ports = list(enumerate(np.unique(train['Embarked'])))
    ports_dict = {port: i for i, port in ports}
    train.Embarked = train.Embarked.map(lambda x: ports_dict[x]).astype(int)
    test.Embarked = test.Embarked.map(lambda x: ports_dict[x]).astype(int)

    # Pull out the deck level from the cabin ids
    train['Deck'] = train['Cabin'].str[:1]
    test['Deck'] = test['Cabin'].str[:1]
    decks = list(enumerate(train['Deck'].unique()))
    decks_dict = {deck: i for i, deck in decks}
    train['Deck'] = train['Deck'].map(lambda x: decks_dict[x]).astype(int)
    test['Deck'] = test['Deck'].map(lambda x: decks_dict[x]).astype(int)

    # Impute missing ages using average for passenger's title  # todo: this is a bit crappy, tidy up
    train['Title'] = train['Name'].str.split(', ').str[1].str.split('. ').str[0].str.upper()
    test['Title'] = test['Name'].str.split(', ').str[1].str.split('. ').str[0].str.upper()
    full_df = pd.concat([train, test])
    boy_age = full_df[(full_df['Title'] == 'MASTER')]['Age'].mean()
    girl_age = full_df[(full_df['Title'] == 'MISS')]['Age'].mean()
    man_age = full_df[(full_df['Sex'] == 1) & (full_df['Title'] != 'MASTER')]['Age'].mean()
    woman_age = full_df[(full_df['Sex'] == 0) & (full_df['Title'] != 'MISS')]['Age'].mean()
    print("Boy:", boy_age, "Girl:", girl_age, "Man:", man_age, "Woman:", woman_age)
    train.loc[train['Age'].isnull() & (train['Title'] == 'MASTER'), 'Age'] = boy_age
    train.loc[train['Age'].isnull() & (train['Title'] == 'MISS'), 'Age'] = girl_age
    train.loc[train['Age'].isnull() & (train['Sex'] == 1) & (train['Title'] != 'MASTER'), 'Age'] = man_age
    train.loc[train['Age'].isnull() & (train['Sex'] == 0) & (train['Title'] != 'MISS'), 'Age'] = woman_age
    test.loc[test['Age'].isnull() & (test['Title'] == 'MASTER'), 'Age'] = boy_age
    test.loc[test['Age'].isnull() & (test['Title'] == 'MISS'), 'Age'] = girl_age
    test.loc[test['Age'].isnull() & (test['Sex'] == 1) & (test['Title'] != 'MASTER'), 'Age'] = man_age
    test.loc[test['Age'].isnull() & (test['Sex'] == 0) & (test['Title'] != 'MISS'), 'Age'] = woman_age

    # Replace null fare value in test data with median for class in all data
    if len(test.Fare[test.Fare.isnull()]) > 0:
        median_fare = np.zeros(3)
        for f in range(0, 3):  # loop 0 to 2
            median_fare[f] = full_df[full_df['Pclass'] == f + 1]['Fare'].dropna().median()
        for f in range(0, 3):
            test.loc[(test.Fare.isnull()) & (test.Pclass == f + 1), 'Fare'] = median_fare[f]

    # drop remaining useless data
    train = train.drop(['Name', 'Cabin', 'Title'], axis=1)
    test = test.drop(['Name', 'Cabin', 'Title'], axis=1)

    return train, test

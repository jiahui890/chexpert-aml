from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier
import numpy as np


def rf_grid():
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {
        'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'bootstrap': bootstrap
    }
    return random_grid


def sgd_grid():
    return {
        'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],  # learning rate
        'loss': ['log'],  # logistic regression,
        'penalty': ['l1', 'l2', 'elasticnet'],
    }


def sgd_grid2():
    return {
        'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],  # learning rate
        'loss': ['log'],  # logistic regression,
        'penalty': ['l1', 'l2', 'elasticnet'],
        'class_weight': [None, 'balanced']
    }


models = {
    "MultinomialNB":
    MultinomialNB(),
    "GaussianNB":
    GaussianNB(),
    "RandomForestClassifier":
    RandomForestClassifier(random_state=42),
    "SGDClassifier":
    SGDClassifier(loss='log', random_state=42),
    "SGDClassifier_Elastic":
    SGDClassifier(loss='huber',
                  penalty='elasticnet',
                  n_jobs=-1,
                  class_weight='balanced')
}

param_grids = {
    "RandomForestClassifier": rf_grid(),
    "SGDClassifier": sgd_grid(),
}
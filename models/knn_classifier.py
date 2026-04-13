from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

def train_knn_classifier(X_train, y_train):
    model = KNeighborsClassifier()

    param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=10,
        scoring='roc_auc',
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_

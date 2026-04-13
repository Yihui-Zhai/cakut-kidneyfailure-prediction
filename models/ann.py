from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTENC
import numpy as np
import warnings

def train_ann_classifier(X_train, y_train):
    if not isinstance(X_train, np.ndarray):
        X_train = X_train.to_numpy()

    n_cv = min(10, int(np.bincount(y_train.astype(int)).min()))

    model = MLPClassifier(
        early_stopping=True,
        validation_fraction=0.125,
        n_iter_no_change=10,
        random_state=42,
        max_iter=200
    )

    param_grid = {
        'hidden_layer_sizes': [(50,), (100,)],
        'alpha': [0.0001, 0.001],
        'learning_rate_init': [0.001, 0.01]
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=n_cv,
        scoring='roc_auc',
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_

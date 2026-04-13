from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

def train_catboost_classifier(X_train, y_train):
    num_pos = np.sum(y_train == 1)
    num_neg = np.sum(y_train == 0)
    scale_pos_weight = num_neg / num_pos

    param_grid = {
        'iterations': [50, 100, 200],
        'depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'l2_leaf_reg': [1, 3, 5, 10],
        'border_count': [32, 50, 100],
    }

    base_model = CatBoostClassifier(
        loss_function='Logloss',
        scale_pos_weight=scale_pos_weight,
        random_seed=42,
        verbose=0,
        task_type='CPU',
    )

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=5,
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print(best_params)

    return best_model

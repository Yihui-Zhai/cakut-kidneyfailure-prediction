from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

def train_gbm_classifier(X_train, y_train):
    num_pos = np.sum(y_train == 1)
    num_neg = np.sum(y_train == 0)

    class_weight = {0: 1.0, 1: num_neg / num_pos}
    n_cv = min(10, int(min(num_pos, num_neg)))

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'max_features': [0.8, 1.0],
    }

    base_model = GradientBoostingClassifier(
        loss='log_loss',
        random_state=42
    )

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=n_cv,
        scoring='roc_auc',
        verbose=1,
        n_jobs=-1
    )

    sample_weight = np.array([class_weight[label] for label in y_train])

    grid_search.fit(X_train, y_train, sample_weight=sample_weight)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print(best_params)

    return best_model

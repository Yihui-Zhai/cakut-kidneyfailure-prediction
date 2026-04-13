from xgboost import XGBClassifier
import numpy as np
from sklearn.model_selection import GridSearchCV

def train_xgboost_classifier(X_train, y_train):
    num_pos = np.sum(y_train == 1)
    num_neg = np.sum(y_train == 0)
    scale_pos_weight = num_neg / num_pos

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [1, 5, 10],
    }
    
    base_model = XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        random_state=42
    )

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=10,
        scoring='roc_auc',
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    print(best_params)

    return best_model
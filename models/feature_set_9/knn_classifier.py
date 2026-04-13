from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTENC
import optuna
from sklearn.model_selection import cross_val_score
import numpy as np

def train_knn_classifier(X_train, y_train, categorical_features=None):
   
    model = KNeighborsClassifier()

    param_grid = {
        'knn__n_neighbors': [3, 5, 7, 9],
        'knn__weights': ['uniform', 'distance'],
        'knn__p': [1, 2]  
    }
    sm=SMOTENC(random_state=42,categorical_features=categorical_features)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('smote',sm),
        ('knn', model)
    ])

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=10,
        scoring='roc_auc',
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_

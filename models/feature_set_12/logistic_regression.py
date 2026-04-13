from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


def train_logistic_regression_classifier(X_train, y_train):
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="l2",
                    solver="lbfgs",
                    class_weight="balanced",
                    random_state=42,
                    max_iter=5000,
                ),
            ),
        ]
    )

    param_grid = {
        "clf__C": [0.01, 0.1, 1.0, 10.0, 100.0],
    }

    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=10,
        scoring="roc_auc",
        verbose=1,
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(grid_search.best_params_)

    return best_model

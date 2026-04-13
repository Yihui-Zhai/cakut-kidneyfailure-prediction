from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import optuna
from sklearn.model_selection import cross_val_score
import warnings

def train_svm_classifier(X_train, y_train):
    def objective(trial):
        param={
            'C':trial.suggest_float('C',1e-2,1e3,log=True),
            'gamma':trial.suggest_float('gamma',1e-4,1.0,log=True),
            'kernel':'rbf'
        }
        model=SVC(
            probability=True,
            class_weight='balanced',
            random_state=42,
            **param
        )
        pipeline=Pipeline([
            ('scaler',StandardScaler()),
            ('svm',model)
        ])

        score =cross_val_score(pipeline,X_train,y_train,cv=5,scoring='roc_auc').mean()
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    best_params = study.best_params
    print(f"BEST PARAMS:{best_params}")

    final_model=SVC(
        probability=True,
        class_weight='balanced',
        random_state=42,
        **best_params
    )

    final_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', final_model)
    ])
    final_pipeline.fit(X_train,y_train)
    return final_pipeline

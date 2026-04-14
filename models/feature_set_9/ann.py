from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
import optuna
from sklearn.model_selection import cross_val_score
import numpy as np
import warnings

def train_ann_classifier(X_train, y_train,categorical_features=None):
    if not isinstance(X_train, np.ndarray):
        X_train = X_train.to_numpy()

    y_train = np.asarray(y_train, dtype=int)
    class_counts = np.bincount(y_train)
    min_class_count = int(class_counts.min())
    n_cv = max(2, min(10, min_class_count))

    # Keep SMOTENC valid inside each training fold for small samples.
    min_train_minority = min_class_count - int(np.ceil(min_class_count / n_cv))
    if categorical_features is not None and min_train_minority >= 2:
        smote_k = min(5, min_train_minority - 1)
    else:
        smote_k = None

    def build_pipeline(model):
        steps = [('scaler', StandardScaler())]
        if categorical_features is not None and smote_k is not None:
            steps.append(
                (
                    'smote',
                    SMOTENC(
                        random_state=42,
                        categorical_features=categorical_features,
                        k_neighbors=smote_k,
                    ),
                )
            )
        steps.append(('ann', model))
        return Pipeline(steps)

    def objective(trial):
        layers_choice =trial.suggest_categorical('layers_choice',['50','100','50_50','100_50'])
        layers_map={
            '50':(50,),
            '100':(100,),
            '50_50':(50,50),
            '100_50':(100,50)
        }
        hidden_layer_sizes=layers_map[layers_choice]
        param={
            'hidden_layer_sizes':hidden_layer_sizes,
            'alpha':trial.suggest_float('alpha',1e-5,1e-1,log=True),
            'learning_rate_init':trial.suggest_float('learning_rate_init',1e-4,1e-2,log=True)
        }
        model=MLPClassifier(
            early_stopping=True,
            validation_fraction=0.125,
            n_iter_no_change=10,
            random_state=42,
            max_iter=300,
            **param
        )
        pipeline = build_pipeline(model)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                score = cross_val_score(
                    pipeline,
                    X_train,
                    y_train,
                    cv=n_cv,
                    scoring='roc_auc',
                ).mean()
        except Exception:
            return -1.0
        return float(score) if np.isfinite(score) else -1.0

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    completed_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.value is not None
        and np.isfinite(t.value)
        and t.value >= 0
    ]

    if not completed_trials:
        final_model = MLPClassifier(
            early_stopping=True,
            validation_fraction=0.125,
            n_iter_no_change=10,
            random_state=42,
            max_iter=300,
        )
        final_pipeline = build_pipeline(final_model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            final_pipeline.fit(X_train, y_train)
        return final_pipeline

    best_params = dict(study.best_trial.params)
    layer_choice =best_params.pop('layers_choice')
    layers_map = {
        '50': (50,),
        '100': (100,),
        '50_50': (50, 50),
        '100_50': (100, 50)
    }
    best_params['hidden_layer_sizes']=layers_map[layer_choice]

    final_model=MLPClassifier(
        early_stopping=True,
        validation_fraction=0.125,
        n_iter_no_change=10,
        random_state=42,
        max_iter=300,
        **best_params
    )
    final_pipeline = build_pipeline(final_model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_pipeline.fit(X_train, y_train)
    return final_pipeline

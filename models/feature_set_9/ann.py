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
        sm = SMOTENC(random_state=42, categorical_features=categorical_features)
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('smote',sm),
            ('ann', model)
        ])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score=cross_val_score(pipeline,X_train,y_train,cv=5,scoring='roc_auc').mean()
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    best_params = study.best_params
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
    final_sm=SMOTENC(random_state=42,categorical_features=categorical_features)
    final_pipeline=Pipeline([
        ('scaler', StandardScaler()),
        ('smote',final_sm),
        ('ann', final_model)
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_pipeline.fit(X_train,y_train)
    return final_pipeline

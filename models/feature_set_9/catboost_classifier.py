from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import optuna
from sklearn.model_selection import cross_val_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTENC
from sklearn.metrics import make_scorer,f1_score

def train_catboost_classifier(X_train, y_train,categorical_features=None):



    min_samples_total=np.min(np.unique(y_train,return_counts=True)[1])
    smote_k_final=min(3,min_samples_total-1)
    def objective(trial):
        param ={
            'iterations': trial.suggest_int('iterations',100,1000),
            'depth': trial.suggest_int('depth',3,7),
            'learning_rate':trial.suggest_float('learning_rate',0.01,0.3,log=True),
            'subsample':trial.suggest_float('subsample',0.7,1.0),
            'l2_leaf_reg':trial.suggest_float('l2_leaf_reg',1,10.0,log=True),
            'border_count':trial.suggest_categorical('border_count',[32,64,128]),
            'loss_function':'Logloss',
            #'scale_pos_weight':scale_pos_weight,
            'random_state':42,
            'verbose':0,
            'task_type':'CPU',
            'thread_count':-1
        }
       
        model= CatBoostClassifier(**param)
        if smote_k_final<=0:
            print("warning:NOT enough minority samples for SMOTE")
            pipeline=Pipeline([('catboost',model)])
        else:
            sm = SMOTENC(random_state=42, categorical_features=categorical_features,k_neighbors=smote_k_final)
            pipeline = Pipeline([
                ('smote', sm),
                ('catboost', model)
            ])
        score =cross_val_score(pipeline,X_train,y_train,cv=5,scoring='roc_auc').mean()
        return score

   
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study =optuna.create_study(direction='maximize')
    study.optimize(objective,n_trials=100)
    if len(study.trials)==0 or all(t.state != optuna.trial.TrialState.COMPLETE for t in study.trials):
        print("no best found")
        base_model=CatBoostClassifier(loss_function='Logloss',
        #scale_pos_weight=scale_pos_weight,
        random_seed=42,
        verbose=0,
        task_type='CPU',
        thread_count=-1)
        if smote_k_final<=0:
            return Pipeline([('catboost',base_model)]).fit(X_train,y_train)
        else:
            sm = SMOTENC(random_state=42, categorical_features=categorical_features,k_neighbors=smote_k_final)
            return  Pipeline([('smote',sm),('catboost',base_model)]).fit(X_train,y_train)
    best_params=study.best_params
    print(f"BEST PARAMS:{best_params}")
    final_model=CatBoostClassifier(
        loss_function='Logloss',
        #scale_pos_weight=scale_pos_weight,
        random_seed=42,
        verbose=0,
        task_type='CPU',
        thread_count=-1,
        **best_params
    )


    if smote_k_final <= 0:
        final_pipeline=Pipeline([('catboost', final_model)])
    else:
        final_sm = SMOTENC(random_state=42, categorical_features=categorical_features, k_neighbors=smote_k_final)
        final_pipeline=Pipeline([('smote', final_sm), ('catboost', final_model)]).fit(X_train, y_train)
    final_pipeline.fit(X_train,y_train)
    return final_pipeline

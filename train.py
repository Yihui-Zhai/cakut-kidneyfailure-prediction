#!/usr/bin/env python

import importlib
import os

import joblib

from utils.data_loading import load_dataset
from config.pipeline import (
    DEFAULT_FEATURE_SETTING,
    DEFAULT_MODEL_LIST,
    DEFAULT_TRAIN_DATASET,
    DEFAULT_YEARS,
    model_dir,
)

TRAINER_SPECS = {
    "lr": ("models.logistic_regression", "train_logistic_regression_classifier"),
    "xgb": ("models.xgb", "train_xgboost_classifier"),
    "rf": ("models.rf", "train_rf_classifier"),
    "svm": ("models.svm", "train_svm_classifier"),
    "knn": ("models.knn_classifier", "train_knn_classifier"),
    "ann": ("models.ann", "train_ann_classifier"),
    "catboost": ("models.catboost_classifier", "train_catboost_classifier"),
    "gbm": ("models.gbm", "train_gbm_classifier"),
    "adaboost": ("models.adaboost", "train_adaboost_classifier"),
}


def load_trainer(model_name):
    if model_name not in TRAINER_SPECS:
        raise KeyError(f"Unknown model: {model_name}")
    module_name, function_name = TRAINER_SPECS[model_name]
    module = importlib.import_module(module_name)
    return getattr(module, function_name)

setting = DEFAULT_FEATURE_SETTING
if __name__ == "__main__":
    training_dataset = DEFAULT_TRAIN_DATASET
    model_storage_dir = model_dir(setting)
    os.makedirs(model_storage_dir, exist_ok=True)

    for year in DEFAULT_YEARS:
        for model_name in DEFAULT_MODEL_LIST:
            try:
                train_method = load_trainer(model_name)
            except Exception as e:
                print(f"[Skip] failed to load trainer model={model_name} ({type(e).__name__}: {e})")
                continue

            try:
                X, y = load_dataset(training_dataset, year, setting=setting)
                best_model = train_method(X, y)
                model_path = f"{model_storage_dir}/{model_name}_{year}yr.pkl"
                joblib.dump(best_model, model_path)
                print(f"Saved: {model_path}")
            except Exception as e:
                print(
                    f"[Skip] failed to train model={model_name} year={year} "
                    f"({type(e).__name__}: {e})"
                )
                continue
            finally:
                print("*" * 50)

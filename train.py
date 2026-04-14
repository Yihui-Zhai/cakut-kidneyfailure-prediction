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

TRAINER_SPECS_12 = {
    "lr": (
        "models.feature_set_12.logistic_regression",
        "train_logistic_regression_classifier",
    ),
    "xgb": ("models.feature_set_12.xgb", "train_xgboost_classifier"),
    "rf": ("models.feature_set_12.rf", "train_rf_classifier"),
    "svm": ("models.feature_set_12.svm", "train_svm_classifier"),
    "knn": ("models.feature_set_12.knn_classifier", "train_knn_classifier"),
    "ann": ("models.feature_set_12.ann", "train_ann_classifier"),
    "catboost": (
        "models.feature_set_12.catboost_classifier",
        "train_catboost_classifier",
    ),
    "gbm": ("models.feature_set_12.gbm", "train_gbm_classifier"),
    "adaboost": ("models.feature_set_12.adaboost", "train_adaboost_classifier"),
}

TRAINER_SPECS_9 = {
    "lr": (
        "models.feature_set_9.logistic_regression",
        "train_logistic_regression_classifier",
    ),
    "xgb": ("models.feature_set_9.xgb", "train_xgboost_classifier"),
    "rf": ("models.feature_set_9.rf", "train_rf_classifier"),
    "svm": ("models.feature_set_9.svm", "train_svm_classifier"),
    "knn": ("models.feature_set_9.knn_classifier", "train_knn_classifier"),
    "ann": ("models.feature_set_9.ann", "train_ann_classifier"),
    "catboost": (
        "models.feature_set_9.catboost_classifier",
        "train_catboost_classifier",
    ),
    "gbm": ("models.feature_set_9.gbm", "train_gbm_classifier"),
    "adaboost": ("models.feature_set_9.adaboost", "train_adaboost_classifier"),
}

MODELS_REQUIRING_CAT_FEATURES_9 = {"knn", "ann", "catboost"}
NUMERICAL_CATEGORIES_9 = {"age_first_diagnose"}


def get_trainer_specs(setting):
    if setting == "9features":
        return TRAINER_SPECS_9
    if setting.startswith("12features"):
        return TRAINER_SPECS_12
    raise KeyError(f"Unsupported feature setting: {setting}")


def load_trainer(model_name, setting):
    trainer_specs = get_trainer_specs(setting)
    if model_name not in trainer_specs:
        raise KeyError(f"Unknown model: {model_name}")
    module_name, function_name = trainer_specs[model_name]
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
                train_method = load_trainer(model_name, setting)
            except Exception as e:
                print(f"[Skip] failed to load trainer model={model_name} ({type(e).__name__}: {e})")
                continue

            try:
                X, y = load_dataset(training_dataset, year, setting=setting)
                if setting == "9features" and model_name in MODELS_REQUIRING_CAT_FEATURES_9:
                    categorical_features = [
                        i for i, col in enumerate(X.columns)
                        if col not in NUMERICAL_CATEGORIES_9
                    ]
                    best_model = train_method(X, y, categorical_features)
                else:
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

"""
Shared pipeline configuration for train/predict/eval scripts.
"""

from __future__ import annotations

from typing import Final

from config.feature_sets import FeatureSetting

DEFAULT_FEATURE_SETTING: Final[FeatureSetting] = "12features"
DEFAULT_YEARS: Final[list[int]] = [1, 3, 5]
DEFAULT_MODEL_LIST: Final[list[str]] = [
    "lr",
    "xgb",
    "rf",
    "svm",
    "knn",
    "ann",
    "catboost",
    "gbm",
    "adaboost",
]

DEFAULT_TRAIN_DATASET: Final[str] = "./dataset/train_example.csv"
DEFAULT_INFER_DATASET: Final[str] = "./dataset/test_example.csv"

MODEL_ARTIFACT_ROOT: Final[str] = "./models/artifacts"
REPORT_ROOT: Final[str] = "./output"


def model_dir(setting: FeatureSetting = DEFAULT_FEATURE_SETTING) -> str:
    return f"{MODEL_ARTIFACT_ROOT}/{setting}"


def report_dir(setting: FeatureSetting = DEFAULT_FEATURE_SETTING) -> str:
    return f"{REPORT_ROOT}/{setting}"

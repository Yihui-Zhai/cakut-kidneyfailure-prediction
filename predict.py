#!/usr/bin/env python
# coding: utf-8

import os
import sys

sys.path.append(os.path.abspath(".."))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config.feature_sets import OUTCOME_COLS, get_features
from config.pipeline import (
    DEFAULT_FEATURE_SETTING,
    DEFAULT_INFER_DATASET,
    DEFAULT_MODEL_LIST,
    DEFAULT_YEARS,
    model_dir,
    report_dir,
)
from utils.data_loading import load_dataset


model_list = DEFAULT_MODEL_LIST
setting = DEFAULT_FEATURE_SETTING
years = DEFAULT_YEARS


def resolve_model_path(model_name, year):
    model_storage_dir = model_dir(setting)
    return f"{model_storage_dir}/{model_name}_{year}yr.pkl"


def predict_score(model, X):
    """Return positive-class score/probability for each sample."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    if hasattr(model, "decision_function"):
        decision = model.decision_function(X)
        decision = np.asarray(decision).reshape(-1)
        # Convert decision value to (0, 1) for easier interpretation.
        return 1.0 / (1.0 + np.exp(-decision))

    return np.asarray(model.predict(X), dtype=float)


def build_prediction_table(dataset_path, keep_test_only=False):
    feature_cols = get_features(setting)
    base_df = pd.read_csv(dataset_path)
    has_all_outcomes = all(col in base_df.columns for col in OUTCOME_COLS)

    keep_cols = [c for c in feature_cols + OUTCOME_COLS if c in base_df.columns]
    result_df = base_df[keep_cols].copy()
    result_df.insert(0, "sample_id", result_df.index)

    # Keep labels visible but map unknown label -1 to NaN.
    for col in OUTCOME_COLS:
        if col in result_df.columns:
            result_df[col] = result_df[col].replace(-1, np.nan)

    for model_name in model_list:
        for year in years:
            result_df[f"{model_name}_pred_{year}y"] = np.nan

    used_indices = set()

    for year in years:
        if has_all_outcomes:
            X_year, y_year = load_dataset(dataset_path, year, setting=setting)
            if keep_test_only:
                _, X_pred, _, y_pred = train_test_split(
                    X_year,
                    y_year,
                    test_size=0.3,
                    random_state=42,
                    stratify=y_year,
                )
            else:
                X_pred, y_pred = X_year, y_year
        else:
            if keep_test_only:
                raise ValueError("keep_test_only=True requires outcome label columns.")
            X_pred = base_df[feature_cols].copy()
            y_pred = None

        valid_idx = X_pred.index
        used_indices.update(valid_idx.tolist())

        for model_name in model_list:
            model_path = resolve_model_path(model_name, year)
            if not os.path.exists(model_path):
                print(f"[Skip] model file not found: {model_path}")
                continue

            try:
                model = joblib.load(model_path)
                pred_scores = predict_score(model, X_pred)
                result_df.loc[valid_idx, f"{model_name}_pred_{year}y"] = pred_scores
            except Exception as e:
                print(
                    f"[Skip] failed model={model_name} year={year} "
                    f"({type(e).__name__}: {e})"
                )
                continue

        if has_all_outcomes:
            # Ensure label column is synchronized with filtered y from load_dataset.
            result_df.loc[valid_idx, f"esrd_{year}y"] = y_pred.values

    if keep_test_only and has_all_outcomes:
        result_df = result_df.loc[sorted(used_indices)].copy()

    return result_df


if __name__ == "__main__":
    infer_dataset = DEFAULT_INFER_DATASET

    report_output_dir = report_dir(setting)
    os.makedirs(report_output_dir, exist_ok=True)

    predictions_df = build_prediction_table(infer_dataset)
    predictions_df.to_excel(f"{report_output_dir}/predictions.xlsx", index=False)

    print(f"Saved: {report_output_dir}/predictions.xlsx")

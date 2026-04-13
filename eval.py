#!/usr/bin/env python
# coding: utf-8


import sys
import os
sys.path.append(os.path.abspath('..'))

from utils.data_loading import load_dataset
import joblib
import pandas as pd
from config.pipeline import (
    DEFAULT_FEATURE_SETTING,
    DEFAULT_INFER_DATASET,
    DEFAULT_MODEL_LIST,
    DEFAULT_YEARS,
    model_dir,
    report_dir,
)
from utils.eval import eval_model

model_list = DEFAULT_MODEL_LIST
setting = DEFAULT_FEATURE_SETTING


def resolve_model_path(model_name, year):
    model_storage_dir = model_dir(setting)
    return f"{model_storage_dir}/{model_name}_{year}yr.pkl"


if __name__ == "__main__":
    eval_dataset = DEFAULT_INFER_DATASET
    report_output_dir = report_dir(setting)
    os.makedirs(report_output_dir, exist_ok=True)

    results = []
    for year in DEFAULT_YEARS:
        X, y = load_dataset(eval_dataset, year, setting=setting)
        
        for model_name in model_list:
            model_path = resolve_model_path(model_name, year)
            try:
                selected_model = joblib.load(model_path)
            except Exception as e:
                print(f"[Skip] failed to load model: {model_path} ({type(e).__name__}: {e})")
                continue
            
            try:
                metrics = eval_model(selected_model, X, y)
            except Exception as e:
                print(f"[Skip] failed to eval model={model_name} year={year} ({type(e).__name__}: {e})")
                continue
            
            metrics['Model'] = model_name
            metrics['Year'] = year
            results.append(metrics)

    results_df = pd.DataFrame(results)
    results_df.to_excel(f"{report_output_dir}/eval.xlsx", index=False)
    print(f"Saved: {report_output_dir}/eval.xlsx")
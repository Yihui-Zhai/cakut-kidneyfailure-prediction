# Predicting Outcome in Children with CAKUT

This repository contains the code for the paper:

> **Interpretable machine learning model for predicting kidney failure among CAKUT children in multicenter large-scale study**

---

## Model Card

### 1. Model Details

| Field | Value |
|---|---|
| **Model Name** | Predicting Outcome in Children with CAKUT (POCC) |
| **Model Type** | XGBoost / Random Forest / Support Vector Machine / K-Nearest Neighbors / Artificial Neural Network / Categorical Boosting / Gradient Boosting Machine / Adaptive Boosting / Logistic Regression classifier |
| **Version** | 1.0 |
| **Date** | April 2026 |
| **Developed by** | Department of Nephrology, Children's Hospital of Fudan University, National Children's Medical Center, Shanghai, China; Shanghai Kidney Development and Pediatric Kidney Disease Research Center, Shanghai, China |
| **Code License** | CC BY-NC 4.0 |
| **Contact** | yhzhai@fudan.edu.cn |

### 2. Intended Use

**Primary intended use:** This model (general and specialized versions) is intended to be used as a clinical decision support tool to predict kidney failure risk at 1, 3, and 5 years post-diagnosis for pediatric patients with congenital anomalies of the kidney and urinary tract (CAKUT).

**Primary intended users:** Pediatric nephrologists, urologists, and pediatricians; pediatric patients with CAKUT at risk of renal progression and their families; and clinical researchers.

**Out-of-scope uses:** This model is not intended for automated, unsupervised prognostic or therapeutic decision-making. It should not be used in adult populations or in pediatric patients with kidney failure caused by non-CAKUT etiologies (e.g., glomerulonephritis).

### 3. Limitations

- **Retrospective nature and information variation:** The model was developed using a long-term retrospective cohort. This may introduce variations in information collection over time, potentially affecting the consistency of data and limiting the model's generalizability in certain historical contexts.

- **Missing clinical variables and longitudinal data:** Due to the multi-center nature of the data, some critical clinical variables (e.g., proteinuria and hypertension) were not captured for all patients. Additionally, the lack of systematically recorded serial longitudinal follow-up data limits the model's ability to account for dynamic clinical changes.

- **Limited real-world validation size:** Although the model underwent real-world validation, the current cohort size is relatively small (n=54). These results should be considered preliminary, and further validation in larger, more diverse real-world populations is required to confirm its robustness.

- **Creatinine-based eGFR limitation:** The model relies on serum creatinine-based eGFR estimates. Since CAKUT patients may have decreased muscle mass, this could lead to an overestimation of kidney function.

- **Model interpretability and clinical application:** While the integration of the SHAP algorithm enhances interpretability and clinician trust, the model remains a supportive tool. AI-generated insights should supplement clinical judgment rather than replace it.

### 4. Potential Fairness Considerations

- **Demographic fairness:** Because the current cohorts are demographically homogeneous, further validation in more diverse racial and ethnic populations is needed.

---

## End-to-end Pipeline

Run the following steps in order.

### Environment Setup

Use the same Python environment for training/inference/evaluation to avoid model
serialization compatibility issues.

Recommended setup:

```bash
conda activate cakut
pip install -r requirements.txt
```

### Step 1: Train models

```bash
python train.py
```

What it does:

- Loads training data from `DEFAULT_TRAIN_DATASET`
- Trains each model in `DEFAULT_MODEL_LIST` for each year in `DEFAULT_YEARS`
- Saves trained model files (`.pkl`)

Important note about skipped models:

- Training may print `[Skip]` for some model/year combinations when data is
  insufficient for the configured cross-validation folds.
- A common case is:
  `ValueError: n_splits=10 cannot be greater than the number of members in each class.`
- This means the script continues running, but that model/year artifact is not
  newly trained in this run.

Saved path:

- `./models/artifacts/{DEFAULT_FEATURE_SETTING}/{model_name}_{year}yr.pkl`
- Example: `./models/artifacts/12features/lr_1yr.pkl`

### Step 2: Generate predictions

```bash
python predict.py
```

What it does:

- Loads inference data from `DEFAULT_INFER_DATASET`
- Loads trained models from `./models/artifacts/{DEFAULT_FEATURE_SETTING}`
- Label columns (`esrd_1y`, `esrd_3y`, `esrd_5y`) are optional for inference
- Generates probability predictions for each model and year
- Exports one table containing features, labels, and prediction columns

Saved path:

- `./output/{DEFAULT_FEATURE_SETTING}/predictions.xlsx`
- Example: `./output/12features/predictions.xlsx`

### Step 3: Evaluate model performance

```bash
python eval.py
```

What it does:

- Loads inference/evaluation data from `DEFAULT_INFER_DATASET`
- Loads trained models from `./models/artifacts/{DEFAULT_FEATURE_SETTING}`
- Computes metrics (AUC, AUPRC, Accuracy, Precision, Recall, etc.)
- Exports an evaluation summary table

Saved path:

- `./output/{DEFAULT_FEATURE_SETTING}/eval.xlsx`
- Example: `./output/12features/eval.xlsx`

## Quick Start Commands

```bash
python train.py
python predict.py
python eval.py
```

Results tables are saved under:

```bash
./output/{DEFAULT_FEATURE_SETTING}/
```

Model `.pkl` files are saved under:

```bash
./models/artifacts/{DEFAULT_FEATURE_SETTING}/
```

## Expected Data Format

Input data must be a CSV file with the same schema as `dataset/train_example.csv` and `dataset/test_example.csv`.

- **Rows**: one patient per row
- **Columns**: feature columns + three outcome columns (`esrd_1y`, `esrd_3y`, `esrd_5y`)
- **Header**: required, exact column names expected by code

## Configuration

### 1) Feature set configuration

If you want to change available feature sets, edit `config/feature_sets.py`.

### 2) Train / Predict / Eval runtime configuration

All pipeline runtime settings are centralized in `config/pipeline.py`:

- `DEFAULT_FEATURE_SETTING`: feature set key (must exist in `config/feature_sets.py`)
- `DEFAULT_MODEL_LIST`: models to train / predict / evaluate, and their order
- `DEFAULT_YEARS`: forecast horizons (for example `[1, 3, 5]`)
- `DEFAULT_TRAIN_DATASET`: training CSV path
- `DEFAULT_INFER_DATASET`: prediction/evaluation CSV path
- `MODEL_ARTIFACT_ROOT`: where trained `.pkl` model files are stored
- `REPORT_ROOT`: where `predictions.xlsx` and `eval.xlsx` are stored

Example configuration:

```python
DEFAULT_FEATURE_SETTING = "12features"
DEFAULT_MODEL_LIST = ["lr", "xgb", "rf"]
DEFAULT_YEARS = [1, 3, 5]
DEFAULT_TRAIN_DATASET = "./dataset/train_example.csv"
DEFAULT_INFER_DATASET = "./dataset/test_example.csv"
```

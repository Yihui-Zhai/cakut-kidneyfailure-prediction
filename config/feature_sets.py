"""
Feature settings for train/predict/eval.

Notes:
  - Feature constants only contain model input columns (no outcomes).
  - To include outcomes, use columns_for_load_dataset().
"""

from __future__ import annotations

from typing import Final, Literal

FeatureSetting = Literal[
    "12features",
    "12features_without_pax",
    "12features_gene_trioplp",
]

OUTCOME_COLS: Final[list[str]] = ["esrd_1y", "esrd_3y", "esrd_5y"]

# 12features: curated clinical features + single gene PAX2
FEATURES_12FEATURES: Final[list[str]] = [
    "PAX2",
    "age_first_diagnose",
    "behavioral_cognitive_abnormalities (1/0)",
    "cakut_subphenotype",
    "ckd_stage_first_diagnose",
    "congenital_heart_disease (1/0)",
    "family_history (1/0)",
    "gender (1/0)",
    "ocular (1/0)",
    "prenatal_phenotype (1/0)",
    "preterm_birth (1/0)",
    "short_stature (1/0)",
]

# 12features_without_pax: 12features minus PAX2
FEATURES_12FEATURES_WITHOUT_PAX: Final[list[str]] = [
    f for f in FEATURES_12FEATURES if f != "PAX2"
]

# 12features_gene_trioplp: keep same clinical features, replace PAX2 with gene_trioplp
FEATURES_12FEATURES_GENE_TRIOPLP: Final[list[str]] = [
    "gene_trioplp (1/0)" if f == "PAX2" else f for f in FEATURES_12FEATURES
]

FEATURE_SETS: dict[FeatureSetting, list[str]] = {
    "12features": list(FEATURES_12FEATURES),
    "12features_without_pax": list(FEATURES_12FEATURES_WITHOUT_PAX),
    "12features_gene_trioplp": list(FEATURES_12FEATURES_GENE_TRIOPLP),
}


def get_features(setting: FeatureSetting) -> list[str]:
    """Return ordered model input columns for a feature setting."""
    return list(FEATURE_SETS[setting])


def columns_for_load_dataset(setting: FeatureSetting) -> list[str]:
    """Columns for load_dataset: features + three outcome columns."""
    return get_features(setting) + OUTCOME_COLS

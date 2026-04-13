from __future__ import annotations

import pandas as pd

from config.feature_sets import (
    OUTCOME_COLS,
    FeatureSetting,
    columns_for_load_dataset,
)

# Keep consistent with outcome columns for legacy imports:
# `from utils.data_loading import y_cols`
y_cols = list(OUTCOME_COLS)


def load_dataset(
    filepath,
    year,
    cols=None,
    *,
    setting: FeatureSetting | None = None,
):
    """
    Read a CSV file and return (X, y). Samples whose outcome for the
    given year is -1 are removed, and all `esrd_*` outcome columns are
    dropped from X.

    Parameters
    ----------
    filepath : str
    year : int
        1 / 3 / 5, corresponding to esrd_1y / esrd_3y / esrd_5y
    cols : list[str] | None
        Explicit columns to read (must include the target outcome column
        and the other two outcome columns to be dropped). Mutually exclusive
        with `setting`.
    setting : {'12features', '12features_without_pax', '12features_gene_trioplp'} | None
        Use a predefined feature set + outcome columns from
        config/feature_sets.py. Mutually exclusive with `cols`.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
    """
    if cols is not None and setting is not None:
        raise ValueError("Specify only one of `cols` or `setting`.")

    if setting is not None:
        cols = columns_for_load_dataset(setting)

    df = pd.read_csv(filepath)

    if cols is not None:
        df = df[cols]

    y_col = f"esrd_{year}y"
    df = df[df[y_col] != -1]
    for col in OUTCOME_COLS:
        if col == y_col:
            y = df[col]
        df.drop(columns=[col], inplace=True)

    return df, y

"""Exploratory analysis helpers for the salary prediction project."""

from salary_prediction.analysis import (
    cat_summary,
    check_df,
    grab_col_names,
    high_correlated_cols,
    num_summary,
    target_summary_with_cat,
)
from salary_prediction.data import load_dataset
from salary_prediction.preprocessing import (
    check_outlier,
    outlier_thresholds,
    replace_with_thresholds,
)

__all__ = [
    "cat_summary",
    "check_df",
    "check_outlier",
    "grab_col_names",
    "high_correlated_cols",
    "load_dataset",
    "num_summary",
    "outlier_thresholds",
    "replace_with_thresholds",
    "target_summary_with_cat",
]

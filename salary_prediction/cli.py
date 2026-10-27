import argparse
from pathlib import Path

import pandas as pd

from salary_prediction.analysis import (
    cat_summary,
    check_df,
    grab_col_names,
    high_correlated_cols,
    num_summary,
    target_summary_with_cat,
)
from salary_prediction.data import DEFAULT_DATA_PATH, load_dataset
from salary_prediction.preprocessing import check_outlier, replace_with_thresholds


def run_eda(
    dataframe: pd.DataFrame, target: str = "Salary", plots: bool = True
) -> None:
    """Run the project's exploratory analysis and outlier review."""
    if target not in dataframe.columns:
        raise ValueError(
            f"Target column '{target}' was not found in the dataset."
        )

    check_df(dataframe)
    categorical_cols, numeric_cols, _ = grab_col_names(dataframe)

    for column in categorical_cols:
        cat_summary(dataframe, column, plot=plots)

    for column in numeric_cols:
        num_summary(dataframe, column, plot=plots)

    for column in categorical_cols:
        target_summary_with_cat(dataframe, target, column)

    high_correlated_cols(dataframe, plot=plots)

    for column in numeric_cols:
        has_outlier = check_outlier(dataframe, column)
        print(f"{column}: {has_outlier}")
        if has_outlier:
            replace_with_thresholds(dataframe, column)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Explore the Hitters salary dataset."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the input CSV (default: Datasets/hitters.csv).",
    )
    parser.add_argument(
        "--target",
        default="Salary",
        help="Name of the target column (default: Salary).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Print summaries without opening charts.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        dataframe = load_dataset(args.data)
        run_eda(dataframe, target=args.target, plots=not args.no_plots)
    except (FileNotFoundError, pd.errors.ParserError, ValueError) as error:
        parser.error(str(error))
    return 0

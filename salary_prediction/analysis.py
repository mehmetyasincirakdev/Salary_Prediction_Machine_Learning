from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


def check_df(dataframe: pd.DataFrame, head: int = 5) -> None:
    """Print a compact overview of a dataframe."""
    if head < 0:
        raise ValueError("head must be zero or greater")

    print("* * * * * HEAD * * * * *")
    print(dataframe.head(head))
    print("* * * * * TAIL * * * * *")
    print(dataframe.tail(head))
    print("* * * * * SHAPE * * * * *")
    print(dataframe.shape)
    print("* * * * * DTYPES * * * * *")
    print(dataframe.dtypes)
    print("* * * * * NA * * * * *")
    print(dataframe.isnull().sum())
    print("* * * * * QUANTILES * * * * *")
    numeric_data = dataframe.select_dtypes(include="number")
    if not numeric_data.empty:
        print(
            numeric_data.quantile(
                [0, 0.05, 0.50, 0.95, 0.99, 1]
            ).T
        )
    else:
        print("No numeric columns.")


def grab_col_names(
    dataframe: pd.DataFrame, cat_th: int = 10, car_th: int = 20
) -> tuple[list[str], list[str], list[str]]:
    """Classify columns as categorical, numerical, or high-cardinality text."""
    if cat_th < 1:
        raise ValueError("cat_th must be greater than zero")
    if car_th < cat_th:
        raise ValueError("car_th must be greater than or equal to cat_th")

    categorical_cols = [
        column
        for column in dataframe.columns
        if not is_numeric_dtype(dataframe[column].dtype)
        or is_bool_dtype(dataframe[column].dtype)
    ]
    numeric_categorical_cols = [
        column
        for column in dataframe.columns
        if is_numeric_dtype(dataframe[column].dtype)
        and not is_bool_dtype(dataframe[column].dtype)
        and dataframe[column].nunique(dropna=True) < cat_th
    ]
    high_cardinality_cols = [
        column
        for column in categorical_cols
        if dataframe[column].nunique(dropna=True) > car_th
    ]

    categorical_cols.extend(numeric_categorical_cols)
    categorical_cols = [
        column
        for column in categorical_cols
        if column not in high_cardinality_cols
    ]
    numeric_cols = [
        column
        for column in dataframe.columns
        if is_numeric_dtype(dataframe[column].dtype)
        and not is_bool_dtype(dataframe[column].dtype)
        and column not in numeric_categorical_cols
    ]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(categorical_cols)}")
    print(f"num_cols: {len(numeric_cols)}")
    print(f"cat_but_car: {len(high_cardinality_cols)}")
    print(f"num_but_cat: {len(numeric_categorical_cols)}")

    return categorical_cols, numeric_cols, high_cardinality_cols


def cat_summary(
    dataframe: pd.DataFrame, col_name: str, plot: bool = False
) -> None:
    """Print category frequencies and, optionally, display a count plot."""
    counts = dataframe[col_name].value_counts(dropna=False)
    summary = pd.DataFrame(
        {col_name: counts, "Ratio": 100 * counts / len(dataframe)}
    )
    print(summary)
    print("##########################################")

    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt

        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


def num_summary(
    dataframe: pd.DataFrame, numerical_col: str, plot: bool = False
) -> None:
    """Print descriptive statistics and, optionally, display a histogram."""
    quantiles = [
        0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
        0.60, 0.70, 0.80, 0.90, 0.95, 0.99,
    ]
    print(dataframe[numerical_col].describe(percentiles=quantiles).T)

    if plot:
        import matplotlib.pyplot as plt

        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()


def target_summary_with_cat(
    dataframe: pd.DataFrame, target: str, categorical_col: str
) -> None:
    """Print the target mean grouped by a categorical column."""
    summary = dataframe.groupby(categorical_col, dropna=False)[target].mean()
    print(pd.DataFrame({"TARGET_MEAN": summary}), end="\n\n\n")


def high_correlated_cols(
    dataframe: pd.DataFrame, plot: bool = False, corr_th: float = 0.90
) -> list[Any]:
    """Return numeric columns correlated above the requested threshold."""
    if not 0 <= corr_th <= 1:
        raise ValueError("corr_th must be between zero and one")

    numeric_data = dataframe.select_dtypes(include="number")
    correlation = numeric_data.corr()
    upper_triangle = correlation.abs().where(
        np.triu(np.ones(correlation.shape, dtype=bool), k=1)
    )
    correlated_cols = [
        column
        for column in upper_triangle.columns
        if (upper_triangle[column] > corr_th).any()
    ]

    if plot:
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set(rc={"figure.figsize": (15, 15)})
        sns.heatmap(correlation, cmap="RdBu")
        plt.show()

    return correlated_cols

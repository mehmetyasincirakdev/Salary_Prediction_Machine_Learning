import pandas as pd


def outlier_thresholds(
    dataframe: pd.DataFrame,
    col_name: str,
    q1: float = 0.25,
    q3: float = 0.75,
) -> tuple[float, float]:
    """Calculate Tukey lower and upper bounds for a numeric column."""
    if not 0 <= q1 < q3 <= 1:
        raise ValueError("quantiles must satisfy 0 <= q1 < q3 <= 1")

    first_quartile = dataframe[col_name].quantile(q1)
    third_quartile = dataframe[col_name].quantile(q3)
    interquartile_range = third_quartile - first_quartile
    return (
        first_quartile - 1.5 * interquartile_range,
        third_quartile + 1.5 * interquartile_range,
    )


def check_outlier(dataframe: pd.DataFrame, col_name: str) -> bool:
    """Return whether a numeric column contains values outside Tukey bounds."""
    lower_bound, upper_bound = outlier_thresholds(dataframe, col_name)
    outside_bounds = (dataframe[col_name] < lower_bound) | (
        dataframe[col_name] > upper_bound
    )
    return bool(outside_bounds.any())


def replace_with_thresholds(
    dataframe: pd.DataFrame, variable: str
) -> None:
    """Cap a numeric column at its Tukey bounds in place."""
    lower_bound, upper_bound = outlier_thresholds(dataframe, variable)
    dataframe[variable] = dataframe[variable].clip(
        lower=lower_bound, upper=upper_bound
    )

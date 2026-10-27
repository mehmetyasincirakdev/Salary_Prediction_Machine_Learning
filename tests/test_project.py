from contextlib import redirect_stdout
from io import StringIO
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from salary_prediction.analysis import grab_col_names, high_correlated_cols
from salary_prediction.cli import run_eda
from salary_prediction.data import load_dataset
from salary_prediction.preprocessing import (
    check_outlier,
    outlier_thresholds,
    replace_with_thresholds,
)


class AnalysisTests(unittest.TestCase):
    def test_grab_col_names_classifies_each_column(self):
        dataframe = pd.DataFrame(
            {
                "team": ["A", "B", "A", "B"],
                "position_code": [1, 2, 1, 1],
                "salary": [100.0, 200.0, 150.0, 250.0],
                "player_name": ["Alex", "Blair", "Casey", "Drew"],
            }
        )

        categorical, numerical, high_cardinality = grab_col_names(
            dataframe, cat_th=3, car_th=3
        )

        self.assertEqual(categorical, ["team", "position_code"])
        self.assertEqual(numerical, ["salary"])
        self.assertEqual(high_cardinality, ["player_name"])

    def test_high_correlated_cols_uses_numeric_columns(self):
        dataframe = pd.DataFrame(
            {
                "first": [1, 2, 3],
                "second": [2, 4, 6],
                "label": ["a", "b", "c"],
            }
        )

        self.assertEqual(high_correlated_cols(dataframe), ["second"])

    def test_run_eda_completes_without_plots(self):
        dataframe = pd.DataFrame(
            {
                "team": ["A", "A", "B", "B"],
                "years": [1, 2, 3, 4],
                "Salary": [100.0, 120.0, 180.0, 200.0],
            }
        )

        with redirect_stdout(StringIO()):
            run_eda(dataframe, plots=False)


class PreprocessingTests(unittest.TestCase):
    def test_outlier_helpers_detect_and_cap_values(self):
        dataframe = pd.DataFrame({"value": [1, 2, 3, 4, 100]})

        self.assertEqual(outlier_thresholds(dataframe, "value"), (-1.0, 7.0))
        self.assertTrue(check_outlier(dataframe, "value"))
        replace_with_thresholds(dataframe, "value")
        self.assertEqual(dataframe["value"].max(), 7.0)
        self.assertFalse(check_outlier(dataframe, "value"))


class DataTests(unittest.TestCase):
    def test_load_dataset_reads_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            path.write_text("Salary,League\n100,A\n", encoding="utf-8")

            dataframe = load_dataset(path)

        self.assertEqual(
            dataframe.to_dict(orient="records"),
            [{"Salary": 100, "League": "A"}],
        )

    def test_load_dataset_reports_missing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.csv"

            with self.assertRaisesRegex(FileNotFoundError, "Dataset not found"):
                load_dataset(path)


if __name__ == "__main__":
    unittest.main()

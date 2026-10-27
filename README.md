# Salary Prediction

This project currently provides exploratory data analysis (EDA) and basic
outlier capping for the Hitters salary dataset. It does not train or persist a
salary prediction model.

## Project layout

```text
salary_prediction/
    analysis.py       # Dataset summaries, column types, and correlations
    cli.py            # EDA workflow and command-line arguments
    data.py           # CSV loading and default dataset location
    preprocessing.py  # Outlier bounds and capping
    __main__.py       # python -m entry point
tests/
    test_project.py
Datasets/
    hitters.csv       # Add the dataset locally; it is not included
```

## Setup

Use Python 3.10 or newer, then install the project dependencies:

```powershell
python -m pip install -r requirements.txt
```

Place the Hitters CSV at `Datasets/hitters.csv`, or provide its path when
running the analysis:

```powershell
python -m salary_prediction
python -m salary_prediction --data "C:\data\hitters.csv" --no-plots
python -m salary_prediction --target Salary --no-plots
```

The CSV must include the selected target column (by default, `Salary`). Charts
are shown by default; `--no-plots` is suitable for non-interactive runs.

## Tests

```powershell
python -m unittest discover -s tests -v
```

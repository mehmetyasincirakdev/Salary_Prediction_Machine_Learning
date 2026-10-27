from pathlib import Path

import pandas as pd

DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parents[1] / "Datasets" / "hitters.csv"
)


def load_dataset(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load a CSV dataset, reporting a useful error when it is unavailable."""
    data_path = Path(path).expanduser()
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. "
            "Provide a CSV with --data or place it at Datasets/hitters.csv."
        )
    return pd.read_csv(data_path)

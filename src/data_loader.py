"""
data_loader.py
──────────────
Handles loading and basic inspection of the student performance dataset.
"""

import pandas as pd
from pathlib import Path

# Canonical path to the raw dataset
RAW_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "student-mat.csv"


def load_data(path: str | Path = RAW_DATA_PATH, sep: str = ";") -> pd.DataFrame:
    """
    Load the Student Performance dataset into a DataFrame.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file. Defaults to data/raw/student-mat.csv.
    sep : str
        Delimiter used in the CSV file (';' for this dataset).

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(path, sep=sep)
    print(f"✅ Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def quick_inspect(df: pd.DataFrame) -> None:
    """Print a quick summary of the dataset."""
    print("\n── Shape ──────────────────────────────")
    print(df.shape)

    print("\n── First 5 rows ────────────────────────")
    print(df.head())

    print("\n── Dtypes ──────────────────────────────")
    print(df.dtypes)

    print("\n── Missing values ───────────────────────")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.any() else "No missing values ✅")

    print("\n── Numeric summary ──────────────────────")
    print(df.describe())


if __name__ == "__main__":
    df = load_data()
    quick_inspect(df)

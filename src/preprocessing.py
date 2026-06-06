"""
preprocessing.py
────────────────
Cleans the raw dataset and engineers features for modelling.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

PROCESSED_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "student_clean.csv"

# Categorical columns in the Student Performance dataset
CATEGORICAL_COLS = [
    "school", "sex", "address", "famsize", "Pstatus",
    "Mjob", "Fjob", "reason", "guardian",
    "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic",
]


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label-encode all categorical (object) columns.

    Returns a copy of the DataFrame with encoded columns.
    """
    df = df.copy()
    le = LabelEncoder()
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])
    print(f"✅ Encoded {len(CATEGORICAL_COLS)} categorical columns")
    return df


def create_target(df: pd.DataFrame, threshold: int = 10) -> pd.DataFrame:
    """
    Add a binary target column 'pass' based on final grade G3.

    Parameters
    ----------
    threshold : int
        Minimum G3 score to count as a pass (default 10 out of 20).
    """
    df = df.copy()
    df["pass"] = (df["G3"] >= threshold).astype(int)
    pass_rate = df["pass"].mean() * 100
    print(f"✅ Target created — pass rate: {pass_rate:.1f}%")
    return df


def drop_grade_leakage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop G1 and G2 (intermediate grades) to prevent data leakage
    when predicting G3 from behavioural/demographic features only.
    """
    df = df.drop(columns=["G1", "G2"], errors="ignore")
    print("✅ Dropped G1, G2 to prevent leakage")
    return df


def preprocess(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
      1. Encode categoricals
      2. Create pass/fail target
      3. Drop leakage columns

    Parameters
    ----------
    save : bool
        If True, saves the cleaned DataFrame to data/processed/.
    """
    df = encode_categoricals(df)
    df = create_target(df)
    df = drop_grade_leakage(df)

    if save:
        PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(PROCESSED_DATA_PATH, index=False)
        print(f"✅ Saved cleaned data → {PROCESSED_DATA_PATH}")

    return df

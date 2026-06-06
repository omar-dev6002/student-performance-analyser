"""
model.py
────────
Trains and evaluates ML models on the preprocessed student data.
To be built out in Week 11 (ML math + sklearn).
"""

import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

MODEL_DIR = Path(__file__).resolve().parents[1] / "outputs" / "models"
TARGET_COL = "pass"


def get_features_target(df: pd.DataFrame):
    """Split DataFrame into feature matrix X and target vector y."""
    X = df.drop(columns=[TARGET_COL, "G3"], errors="ignore")
    y = df[TARGET_COL]
    return X, y


def train(df: pd.DataFrame, model_name: str = "random_forest", test_size: float = 0.2):
    """
    Train a classifier and print evaluation metrics.

    Parameters
    ----------
    model_name : str
        'random_forest' or 'logistic_regression'
    test_size : float
        Fraction of data held out for testing.
    """
    X, y = get_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    if model_name == "random_forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == "logistic_regression":
        model = LogisticRegression(max_iter=1000, random_state=42)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n── {model_name} results ─────────────────")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_path = MODEL_DIR / f"{model_name}.pkl"
    joblib.dump(model, save_path)
    print(f"\n✅ Model saved → {save_path}")

    return model


# ── TODO (Week 11–12) ──────────────────────────────────────────────────────────
# - Add cross-validation
# - Add GridSearchCV for hyperparameter tuning
# - Add feature importance plot
# - Try additional models (SVM, KNN, GradientBoosting)

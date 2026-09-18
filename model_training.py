"""
Ridge Regression training and evaluation for Marketing Mix Model (MMM).

Accepts final_data from app.py (date, sales, and channel columns sorted
chronologically), performs an 80/20 chronological train-test split,
scales features, trains Ridge Regression, and reports accuracy metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler

TRAIN_RATIO = 0.8
DEFAULT_RIDGE_ALPHA = 1.0


@dataclass
class TrainingResult:
    """Trained Ridge model and artifacts needed for evaluation."""

    model: Ridge
    scaler: StandardScaler
    feature_columns: list[str]
    date_col: str
    sales_col: str
    train_data: pd.DataFrame
    test_data: pd.DataFrame
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    X_train_scaled: np.ndarray
    X_test_scaled: np.ndarray
    y_train_pred: np.ndarray
    y_test_pred: np.ndarray


def chronological_train_test_split(
    final_data: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split final_data chronologically into training and test sets.

    The first train_ratio fraction of rows (by date order) is used for
    training; the remainder is held out for testing.
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")

    n_rows = len(final_data)
    if n_rows < 2:
        raise ValueError("final_data must contain at least 2 rows.")

    split_idx = int(n_rows * train_ratio)
    split_idx = max(1, min(split_idx, n_rows - 1))

    train_data = final_data.iloc[:split_idx].copy()
    test_data = final_data.iloc[split_idx:].copy()
    return train_data, test_data


def extract_features_and_target(
    data: pd.DataFrame,
    date_col: str,
    sales_col: str,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Separate inputs (X) and sales target (y) from a data slice.

    X contains every column except date and sales.
    """
    if date_col not in data.columns:
        raise ValueError(f"Date column '{date_col}' not found in data.")
    if sales_col not in data.columns:
        raise ValueError(f"Sales column '{sales_col}' not found in data.")

    feature_columns = [col for col in data.columns if col not in (date_col, sales_col)]
    if not feature_columns:
        raise ValueError("No feature columns found (expected channel columns).")

    X = data[feature_columns].copy()
    y = data[sales_col].copy()
    return X, y, feature_columns


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[StandardScaler, np.ndarray, np.ndarray]:
    """Fit a StandardScaler on training features and transform train and test."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return scaler, X_train_scaled, X_test_scaled


def train_ridge_model(
    final_data: pd.DataFrame,
    date_col: str,
    sales_col: str,
    train_ratio: float = TRAIN_RATIO,
    alpha: float = DEFAULT_RIDGE_ALPHA,
) -> TrainingResult:
    """
    Chronologically split final_data, scale features, and train Ridge Regression.

    Returns the trained model and all data needed for evaluation.
    """
    feature_columns = [
        col for col in final_data.columns if col not in (date_col, sales_col)
    ]
    required_cols = [date_col, sales_col] + feature_columns
    clean_data = final_data.dropna(subset=required_cols).reset_index(drop=True)

    if len(clean_data) < 2:
        raise ValueError(
            "final_data must contain at least 2 complete rows after removing missing values."
        )

    train_data, test_data = chronological_train_test_split(clean_data, train_ratio)

    X_train, y_train, feature_columns = extract_features_and_target(
        train_data, date_col, sales_col
    )
    X_test, y_test, _ = extract_features_and_target(test_data, date_col, sales_col)

    scaler, X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    model = Ridge(alpha=alpha)
    model.fit(X_train_scaled, y_train)

    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    return TrainingResult(
        model=model,
        scaler=scaler,
        feature_columns=feature_columns,
        date_col=date_col,
        sales_col=sales_col,
        train_data=train_data,
        test_data=test_data,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        X_train_scaled=X_train_scaled,
        X_test_scaled=X_test_scaled,
        y_train_pred=y_train_pred,
        y_test_pred=y_test_pred,
    )


def compute_regression_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute standard regression accuracy metrics."""
    y_true_arr = np.asarray(y_true)
    metrics: dict[str, float] = {
        "mse": float(mean_squared_error(y_true_arr, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_arr, y_pred))),
        "mae": float(mean_absolute_error(y_true_arr, y_pred)),
        "r2": float(r2_score(y_true_arr, y_pred)),
    }

    # MAPE is undefined when any true value is zero; guard against that.
    if np.any(y_true_arr == 0):
        metrics["mape"] = float("nan")
    else:
        metrics["mape"] = float(mean_absolute_percentage_error(y_true_arr, y_pred))

    return metrics


def evaluate_model(result: TrainingResult) -> dict[str, Any]:
    """
    Evaluate the trained Ridge model on training and test sets.

    Returns a dictionary with metrics and prediction tables.
    """
    train_metrics = compute_regression_metrics(result.y_train, result.y_train_pred)
    test_metrics = compute_regression_metrics(result.y_test, result.y_test_pred)

    test_predictions = pd.DataFrame(
        {
            result.date_col: result.test_data[result.date_col].values,
            "actual_sales": result.y_test.values,
            "predicted_sales": result.y_test_pred,
            "residual": result.y_test.values - result.y_test_pred,
        }
    )

    return {
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "train_size": len(result.train_data),
        "test_size": len(result.test_data),
        "feature_columns": result.feature_columns,
        "ridge_alpha": result.model.alpha,
        "coefficients": dict(zip(result.feature_columns, result.model.coef_)),
        "intercept": float(result.model.intercept_),
        "test_predictions": test_predictions,
    }


def train_and_evaluate(
    final_data: pd.DataFrame,
    date_col: str,
    sales_col: str,
    train_ratio: float = TRAIN_RATIO,
    alpha: float = DEFAULT_RIDGE_ALPHA,
) -> tuple[TrainingResult, dict[str, Any]]:
    """
    Train Ridge Regression on final_data and evaluate on the held-out test set.

    Returns (training_result, evaluation_report).
    """
    result = train_ridge_model(
        final_data,
        date_col=date_col,
        sales_col=sales_col,
        train_ratio=train_ratio,
        alpha=alpha,
    )
    evaluation = evaluate_model(result)
    return result, evaluation


def print_evaluation_report(evaluation: dict[str, Any]) -> None:
    """Print training and test accuracy metrics to stdout."""
    print("=" * 60)
    print("Ridge Regression — Model Evaluation")
    print("=" * 60)
    print(f"Training rows : {evaluation['train_size']}")
    print(f"Test rows     : {evaluation['test_size']}")
    print(f"Features      : {', '.join(evaluation['feature_columns'])}")
    print(f"Ridge alpha   : {evaluation['ridge_alpha']}")
    print()

    print("Training set metrics:")
    for name, value in evaluation["train_metrics"].items():
        print(f"  {name.upper():6s}: {value:,.4f}")

    print()
    print("Test set metrics:")
    for name, value in evaluation["test_metrics"].items():
        print(f"  {name.upper():6s}: {value:,.4f}")

    print()
    print("Model coefficients (scaled features):")
    print(f"  intercept: {evaluation['intercept']:,.4f}")
    for feature, coef in evaluation["coefficients"].items():
        print(f"  {feature}: {coef:,.4f}")

    print()
    print("Test set predictions (first 10 rows):")
    print(evaluation["test_predictions"].head(10).to_string(index=False))
    print("=" * 60)


def _build_demo_final_data(
    collated_df: pd.DataFrame,
    date_col: str,
    sales_col: str,
    channel_cols: list[str],
) -> pd.DataFrame:
    """Mirror app.build_final_data without importing the Streamlit app module."""
    cols = [date_col, sales_col] + channel_cols
    final_data = collated_df[cols].copy()
    final_data[date_col] = pd.to_datetime(final_data[date_col], errors="coerce")
    for col in [sales_col] + channel_cols:
        final_data[col] = pd.to_numeric(final_data[col], errors="coerce")
    return final_data.sort_values(by=date_col, ascending=True).reset_index(drop=True)


if __name__ == "__main__":
    from pathlib import Path

    data_dir = Path(__file__).resolve().parent / "synthetic_data"
    excel_files = sorted(data_dir.glob("*.xlsx"))

    if len(excel_files) < 5:
        print("Run test_streamlit.py first to generate synthetic_data/*.xlsx files.")
    else:
        dataframes = [
            pd.read_excel(f, engine="openpyxl").rename(columns=str.strip)
            for f in excel_files[:5]
        ]
        all_columns = sorted({col for df in dataframes for col in df.columns})
        aligned = [df.reindex(columns=all_columns) for df in dataframes]
        collated_df = pd.concat(aligned, ignore_index=True)

        date_col = "date"
        sales_col = "sales"
        channel_cols = [
            "TV_spend",
            "Meta_spend",
            "Youtube_spend",
            "PR_spend",
            "Instagram_spend",
        ]
        channel_cols = [c for c in channel_cols if c in collated_df.columns]

        final_data = _build_demo_final_data(
            collated_df, date_col, sales_col, channel_cols
        )
        _, evaluation = train_and_evaluate(
            final_data, date_col=date_col, sales_col=sales_col
        )
        print_evaluation_report(evaluation)

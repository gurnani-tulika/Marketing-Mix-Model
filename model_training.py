"""
Ridge Regression training and evaluation for Marketing Mix Model (MMM).

Accepts final_data from app.py (date, sales, and channel columns sorted
chronologically), applies geometric adstock to each marketing channel,
adds Fourier seasonality terms (weekly/monthly when data frequency supports
them), performs an 80/20 chronological train-test split, scales features,
trains Ridge Regression, and reports accuracy metrics.
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
ADSTOCK_DECAY_GRID = np.arange(0.0, 1.0, 0.1)
ADSTOCK_OPTIMIZATION_ROUNDS = 2


@dataclass
class TrainingResult:
    """Trained Ridge model and artifacts needed for evaluation."""

    model: Ridge
    scaler: StandardScaler
    feature_columns: list[str]
    channel_columns: list[str]
    adstock_decays: dict[str, float]
    seasonality_columns: list[str]
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


def geometric_adstock(
    spend: np.ndarray,
    decay: float,
    initial_carryover: float = 0.0,
) -> np.ndarray:
    """
    Apply geometric adstock: adstock[t] = spend[t] + decay * adstock[t - 1].

    initial_carryover is the last adstocked value from a prior period (e.g.
    end of training data when transforming the test set).
    """
    spend_arr = np.asarray(spend, dtype=float)
    adstocked = np.empty_like(spend_arr)
    previous = float(initial_carryover)

    for idx, value in enumerate(spend_arr):
        current = float(value) + decay * previous
        adstocked[idx] = current
        previous = current

    return adstocked


def infer_date_granularity(dates: pd.Series) -> str:
    """Infer whether the series is daily, weekly, or monthly from date gaps."""
    ordered = pd.to_datetime(dates, errors="coerce").sort_values().dropna()
    if len(ordered) < 2:
        return "daily"

    median_gap_days = ordered.diff().dropna().dt.days.median()
    if median_gap_days <= 2:
        return "daily"
    if median_gap_days <= 10:
        return "weekly"
    return "monthly"


def build_seasonality_features(dates: pd.Series) -> pd.DataFrame:
    """
    Build sine/cosine Fourier features for weekly and monthly seasonality.

    Ridge learns the coefficients, so near-zero weights mean little seasonal
    effect. Sin/cos pairs let the model recover amplitude and phase.
    """
    dt = pd.to_datetime(dates, errors="coerce")
    granularity = infer_date_granularity(dt)
    features: dict[str, np.ndarray] = {}

    if granularity == "daily":
        day_of_week = dt.dt.dayofweek.to_numpy(dtype=float)
        features["weekly_sin"] = np.sin(2 * np.pi * day_of_week / 7.0)
        features["weekly_cos"] = np.cos(2 * np.pi * day_of_week / 7.0)

        day_of_month = dt.dt.day.to_numpy(dtype=float)
        days_in_month = dt.dt.days_in_month.to_numpy(dtype=float)
        features["monthly_sin"] = np.sin(2 * np.pi * day_of_month / days_in_month)
        features["monthly_cos"] = np.cos(2 * np.pi * day_of_month / days_in_month)

        day_of_year = dt.dt.dayofyear.to_numpy(dtype=float)
        days_in_year = np.where(dt.dt.is_leap_year, 366.0, 365.0)
        features["yearly_sin"] = np.sin(2 * np.pi * day_of_year / days_in_year)
        features["yearly_cos"] = np.cos(2 * np.pi * day_of_year / days_in_year)
    elif granularity == "weekly":
        week_of_year = dt.dt.isocalendar().week.to_numpy(dtype=float)
        features["yearly_sin"] = np.sin(2 * np.pi * week_of_year / 52.0)
        features["yearly_cos"] = np.cos(2 * np.pi * week_of_year / 52.0)
    else:
        month = dt.dt.month.to_numpy(dtype=float)
        features["yearly_sin"] = np.sin(2 * np.pi * month / 12.0)
        features["yearly_cos"] = np.cos(2 * np.pi * month / 12.0)

    return pd.DataFrame(features, index=dates.index)


def apply_adstock_to_channels(
    data: pd.DataFrame,
    channel_columns: list[str],
    decays: dict[str, float],
    carryover: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Transform raw channel spend columns with per-channel geometric adstock."""
    adstocked = data[channel_columns].copy()
    final_carryover: dict[str, float] = {}

    for channel in channel_columns:
        decay = decays.get(channel, 0.0)
        initial = (carryover or {}).get(channel, 0.0)
        adstocked[channel] = geometric_adstock(
            data[channel].to_numpy(dtype=float),
            decay=decay,
            initial_carryover=initial,
        )
        final_carryover[channel] = float(adstocked[channel].iloc[-1])

    return adstocked, final_carryover


def build_model_features(
    data: pd.DataFrame,
    date_col: str,
    channel_columns: list[str],
    adstock_decays: dict[str, float],
    carryover: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Combine adstocked channel spends with Fourier seasonality features."""
    adstocked_channels, final_carryover = apply_adstock_to_channels(
        data,
        channel_columns,
        adstock_decays,
        carryover=carryover,
    )
    seasonality = build_seasonality_features(data[date_col])
    features = pd.concat([adstocked_channels, seasonality], axis=1)
    return features, final_carryover


def _fit_ridge_mse(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_eval: np.ndarray,
    y_eval: pd.Series,
    alpha: float,
) -> float:
    """Fit Ridge on train features and return MSE on an evaluation slice."""
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    predictions = model.predict(X_eval)
    return float(mean_squared_error(y_eval, predictions))


def select_adstock_decays(
    train_data: pd.DataFrame,
    date_col: str,
    sales_col: str,
    channel_columns: list[str],
    alpha: float,
    decay_grid: np.ndarray = ADSTOCK_DECAY_GRID,
    optimization_rounds: int = ADSTOCK_OPTIMIZATION_ROUNDS,
) -> dict[str, float]:
    """
    Select per-channel adstock decay rates by coordinate search on training MSE.

    Seasonality features are always included so decay selection accounts for
    seasonal structure in sales.
    """
    y_train = train_data[sales_col]
    decays = {channel: 0.5 for channel in channel_columns}
    seasonality = build_seasonality_features(train_data[date_col])

    for _ in range(optimization_rounds):
        for channel in channel_columns:
            best_decay = decays[channel]
            best_mse = np.inf

            for decay in decay_grid:
                trial_decays = decays.copy()
                trial_decays[channel] = float(decay)

                adstocked, _ = apply_adstock_to_channels(
                    train_data, channel_columns, trial_decays
                )
                X = pd.concat([adstocked, seasonality], axis=1)

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                mse = _fit_ridge_mse(X_scaled, y_train, X_scaled, y_train, alpha)
                if mse < best_mse:
                    best_mse = mse
                    best_decay = float(decay)

            decays[channel] = best_decay

    return decays


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
    Chronologically split final_data, apply adstock and seasonality features,
    scale features, and train Ridge Regression.

    Returns the trained model and all data needed for evaluation.
    """
    channel_columns = [
        col for col in final_data.columns if col not in (date_col, sales_col)
    ]
    required_cols = [date_col, sales_col] + channel_columns
    clean_data = final_data.dropna(subset=required_cols).reset_index(drop=True)

    if len(clean_data) < 2:
        raise ValueError(
            "final_data must contain at least 2 complete rows after removing missing values."
        )

    train_data, test_data = chronological_train_test_split(clean_data, train_ratio)

    adstock_decays = select_adstock_decays(
        train_data,
        date_col=date_col,
        sales_col=sales_col,
        channel_columns=channel_columns,
        alpha=alpha,
    )

    X_train, train_carryover = build_model_features(
        train_data,
        date_col=date_col,
        channel_columns=channel_columns,
        adstock_decays=adstock_decays,
    )
    X_test, _ = build_model_features(
        test_data,
        date_col=date_col,
        channel_columns=channel_columns,
        adstock_decays=adstock_decays,
        carryover=train_carryover,
    )

    seasonality_columns = [
        col for col in X_train.columns if col not in channel_columns
    ]
    feature_columns = list(X_train.columns)

    y_train = train_data[sales_col].copy()
    y_test = test_data[sales_col].copy()

    scaler, X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    model = Ridge(alpha=alpha)
    model.fit(X_train_scaled, y_train)

    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    return TrainingResult(
        model=model,
        scaler=scaler,
        feature_columns=feature_columns,
        channel_columns=channel_columns,
        adstock_decays=adstock_decays,
        seasonality_columns=seasonality_columns,
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
        "channel_columns": result.channel_columns,
        "adstock_decays": result.adstock_decays,
        "seasonality_columns": result.seasonality_columns,
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
    print("Adstock decay rates (per channel):")
    for channel, decay in evaluation["adstock_decays"].items():
        print(f"  {channel}: {decay:.2f}")
    print()
    print("Seasonality features:")
    print(f"  {', '.join(evaluation['seasonality_columns'])}")
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

    if not excel_files:
        print("Run test_streamlit.py first to generate synthetic_data/*.xlsx files.")
    else:
        dataframes = [
            pd.read_excel(f, engine="openpyxl").rename(columns=str.strip)
            for f in excel_files[:5]
        ]
        print(f"Using {len(dataframes)} uploaded file(s) from synthetic_data/.")
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

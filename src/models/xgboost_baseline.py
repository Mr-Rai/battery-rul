"""
XGBoost baseline model for battery RUL estimation.

This is Layer 1 of the modeling stack — the interpretable benchmark.
Every more complex model (TCN, Bayesian) must beat this to justify
the added complexity. If XGBoost is within margin of the TCN,
we deploy XGBoost — simpler, faster, more explainable.

Training is logged to MLflow. SHAP values are computed post-training
and saved as artifacts for explainability review.
"""

import logging
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict,
    n_splits: int = 5,
) -> xgb.XGBRegressor:
    """
    Train XGBoost with time-series cross-validation.

    CRITICAL: Standard k-fold must NOT be used on time-series data.
    TimeSeriesSplit ensures no future data leaks into training folds.

    Args:
        X_train: Feature matrix (cycle-level, time-ordered).
        y_train: RUL target (in cycles remaining).
        params: XGBoost hyperparameters from config.
        n_splits: Number of temporal CV folds.

    Returns:
        Fitted XGBRegressor.
    """
    # TODO: implement
    raise NotImplementedError


def evaluate(
    model: xgb.XGBRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Evaluate model and return metrics dict.

    Metrics: RMSE, MAE, MAPE, early warning rate (% cells flagged >20 cycles before EOL).

    Args:
        model: Fitted XGBRegressor.
        X_test: Test feature matrix.
        y_test: True RUL values.

    Returns:
        Dict of metric name -> value.
    """
    # TODO: implement
    raise NotImplementedError


def compute_shap(
    model: xgb.XGBRegressor,
    X: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """
    Compute and save SHAP values for model explainability.

    Saves:
        - shap_summary.png: beeswarm plot
        - shap_values.parquet: raw SHAP matrix for downstream analysis

    Args:
        model: Fitted XGBRegressor.
        X: Feature matrix to explain (typically test set).
        output_dir: Directory to save outputs.
    """
    # TODO: implement
    raise NotImplementedError


def run(config_path: str = "configs/xgb_baseline.yaml") -> None:
    """
    Full training run with MLflow tracking.
    Loads config, trains, evaluates, logs everything to MLflow.
    """
    # TODO: implement
    raise NotImplementedError


if __name__ == "__main__":
    run()

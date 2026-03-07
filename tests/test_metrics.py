"""
Unit tests for evaluation metrics.
These run without any model or data — pure function tests.
"""

import numpy as np
import pytest
from src.evaluation.metrics import rmse, mae, mape, prediction_interval_coverage


def test_rmse_perfect_prediction():
    y = np.array([10.0, 20.0, 30.0])
    assert rmse(y, y) == pytest.approx(0.0)


def test_rmse_known_value():
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([3.0, 4.0])
    # errors: 3, 4 → MSE = (9+16)/2 = 12.5 → RMSE = sqrt(12.5)
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(12.5))


def test_mae_known_value():
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([12.0, 15.0])
    assert mae(y_true, y_pred) == pytest.approx(3.5)


def test_mape_no_division_by_zero():
    y_true = np.array([0.0, 10.0])
    y_pred = np.array([1.0, 12.0])
    result = mape(y_true, y_pred)
    assert np.isfinite(result)


def test_pi_coverage_full():
    y_true = np.array([5.0, 10.0, 15.0])
    lower = np.array([4.0, 9.0, 14.0])
    upper = np.array([6.0, 11.0, 16.0])
    assert prediction_interval_coverage(y_true, lower, upper) == pytest.approx(1.0)


def test_pi_coverage_none():
    y_true = np.array([5.0, 10.0])
    lower = np.array([6.0, 11.0])
    upper = np.array([8.0, 13.0])
    assert prediction_interval_coverage(y_true, lower, upper) == pytest.approx(0.0)

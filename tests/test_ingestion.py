"""
tests/test_ingestion.py

Unit tests for ingestion pipeline components.
Test with: pytest tests/ -v --cov=src
"""

import numpy as np
import pandas as pd
import pytest
from src.ingestion.calce_loader import extract_cycles


def make_mock_raw_df(n_cycles: int = 100, nominal_capacity: float = 1.1,
                     fade_rate: float = 0.003) -> pd.DataFrame:
    """
    Generate synthetic raw battery time-series for testing.
    Simulates linear capacity fade — deterministic and verifiable.
    """
    rows = []
    for cycle in range(1, n_cycles + 1):
        capacity = nominal_capacity - fade_rate * cycle
        capacity = max(capacity, 0.5)
        # Simulate discharge steps (current negative = discharge)
        for step in range(10):
            rows.append({
                "Cycle_Index": cycle,
                "Step_Index": step,
                "Current(A)": -1.0,
                "Voltage(V)": 3.8 - step * 0.02,
                "Discharge_Capacity(Ah)": capacity,
                "Charge_Capacity(Ah)": 0.0,
                "Temperature (C)": 25.0 + np.random.normal(0, 0.1),
            })
    return pd.DataFrame(rows)


class TestExtractCycles:

    def test_output_shape(self):
        df = make_mock_raw_df(n_cycles=100)
        result = extract_cycles(df, nominal_capacity=1.1, eol_threshold=0.80)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_required_columns_present(self):
        df = make_mock_raw_df(n_cycles=100)
        result = extract_cycles(df, nominal_capacity=1.1, eol_threshold=0.80)
        required = ["cycle_index", "discharge_capacity", "soh", "rul"]
        for col in required:
            assert col in result.columns, f"Missing column: {col}"

    def test_soh_bounded(self):
        df = make_mock_raw_df(n_cycles=100)
        result = extract_cycles(df, nominal_capacity=1.1, eol_threshold=0.80)
        assert result["soh"].between(0, 1).all(), "SOH values out of [0, 1] range"

    def test_rul_non_negative(self):
        df = make_mock_raw_df(n_cycles=100)
        result = extract_cycles(df, nominal_capacity=1.1, eol_threshold=0.80)
        assert (result["rul"] >= 0).all(), "RUL contains negative values"

    def test_rul_decreasing(self):
        """RUL must be monotonically non-increasing as cycles progress."""
        df = make_mock_raw_df(n_cycles=200, fade_rate=0.003)
        result = extract_cycles(df, nominal_capacity=1.1, eol_threshold=0.80)
        if result["rul"].notna().all():
            diffs = result["rul"].diff().dropna()
            assert (diffs <= 0).all(), "RUL is not monotonically decreasing"

    def test_eol_cycle_has_zero_rul(self):
        """The cycle at EOL must have RUL = 0."""
        df = make_mock_raw_df(n_cycles=200, fade_rate=0.003)
        result = extract_cycles(df, nominal_capacity=1.1, eol_threshold=0.80)
        if result["rul"].notna().all():
            assert result["rul"].min() == 0, "EOL cycle does not have RUL = 0"

    def test_no_zero_capacity_cycles(self):
        df = make_mock_raw_df(n_cycles=50)
        result = extract_cycles(df, nominal_capacity=1.1, eol_threshold=0.80)
        assert (result["discharge_capacity"] > 0).all()


class TestSOHComputation:

    def test_soh_equals_one_at_start(self):
        """First cycle SOH should be close to 1.0 with no degradation."""
        df = make_mock_raw_df(n_cycles=50, fade_rate=0.0)
        result = extract_cycles(df, nominal_capacity=1.1, eol_threshold=0.80)
        first_soh = result.iloc[0]["soh"]
        assert abs(first_soh - 1.0) < 0.05, f"Expected SOH ~1.0 at start, got {first_soh}"

    def test_soh_decreases_with_fade(self):
        """SOH must decrease as cycles progress under non-zero fade rate."""
        df = make_mock_raw_df(n_cycles=100, fade_rate=0.005)
        result = extract_cycles(df, nominal_capacity=1.1, eol_threshold=0.80)
        assert result["soh"].iloc[0] > result["soh"].iloc[-1], "SOH not decreasing"

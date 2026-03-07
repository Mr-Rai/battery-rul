"""
src/features/feature_engineer.py

Physics-informed and statistical feature engineering for battery RUL estimation.

Two categories of features:

1. PHYSICS-INFORMED:
   - Incremental Capacity Analysis (ICA): dQ/dV vs V
     Peaks in this curve correspond to phase transitions in the electrode.
     As the battery degrades, these peaks shift and diminish.
     Tracking peak position and height captures electrochemical aging.

   - Differential Voltage Analysis (DVA): dV/dQ vs Q
     Complementary to ICA. Valley positions in DVA correspond to
     the same phase transitions seen as peaks in ICA.

2. STATISTICAL (cycle-level):
   - Rolling statistics over past N cycles (capacity fade trend)
   - Internal resistance proxy (delta V / delta I at cycle start)
   - Charge time features
   - Capacity fade rate (slope of SOH over rolling window)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.signal import savgol_filter, find_peaks

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/base.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ─── ICA / DVA (require raw time-series, not cycle-level summaries) ───────────

def compute_ica(voltage: np.ndarray, capacity: np.ndarray, bins: int = 1000) -> dict:
    """
    Compute Incremental Capacity Analysis (dQ/dV) curve and extract peak features.

    Returns dict with:
    - ica_peak_height:    height of the primary dQ/dV peak
    - ica_peak_voltage:   voltage at the primary peak
    - ica_peak_width:     width of the primary peak at half-height
    - ica_area:           area under the dQ/dV curve (integral)
    """
    # Sort by voltage and remove duplicates
    sort_idx = np.argsort(voltage)
    v = voltage[sort_idx]
    q = capacity[sort_idx]

    # Bin voltage range
    v_bins = np.linspace(v.min(), v.max(), bins)
    q_interp = np.interp(v_bins, v, q)

    # Differentiate: dQ/dV
    dqdv = np.gradient(q_interp, v_bins)

    # Smooth to reduce noise
    if len(dqdv) > 11:
        dqdv = savgol_filter(dqdv, window_length=11, polyorder=3)

    # Find peaks
    peaks, properties = find_peaks(dqdv, height=0, width=1)

    features = {
        "ica_peak_height": 0.0,
        "ica_peak_voltage": 0.0,
        "ica_peak_width": 0.0,
        "ica_area": float(np.trapz(np.clip(dqdv, 0, None), v_bins)),
    }

    if len(peaks) > 0:
        # Take the dominant peak (highest)
        dominant = peaks[np.argmax(properties["peak_heights"])]
        features["ica_peak_height"] = float(dqdv[dominant])
        features["ica_peak_voltage"] = float(v_bins[dominant])
        features["ica_peak_width"] = float(properties["widths"][np.argmax(properties["peak_heights"])])

    return features


def compute_dva(voltage: np.ndarray, capacity: np.ndarray, bins: int = 1000) -> dict:
    """
    Compute Differential Voltage Analysis (dV/dQ) curve and extract valley features.

    Returns dict with:
    - dva_valley_depth:   depth of the primary dV/dQ valley
    - dva_valley_capacity: capacity at the primary valley
    - dva_area:           area under the absolute dV/dQ curve
    """
    sort_idx = np.argsort(capacity)
    q = capacity[sort_idx]
    v = voltage[sort_idx]

    q_bins = np.linspace(q.min(), q.max(), bins)
    v_interp = np.interp(q_bins, q, v)

    dvdq = np.gradient(v_interp, q_bins)

    if len(dvdq) > 11:
        dvdq = savgol_filter(dvdq, window_length=11, polyorder=3)

    # Valleys in dV/dQ = peaks in -dV/dQ
    valleys, properties = find_peaks(-dvdq, height=0, width=1)

    features = {
        "dva_valley_depth": 0.0,
        "dva_valley_capacity": 0.0,
        "dva_area": float(np.trapz(np.abs(dvdq), q_bins)),
    }

    if len(valleys) > 0:
        dominant = valleys[np.argmax(properties["peak_heights"])]
        features["dva_valley_depth"] = float(-dvdq[dominant])
        features["dva_valley_capacity"] = float(q_bins[dominant])

    return features


# ─── Cycle-level statistical features ─────────────────────────────────────────

def add_rolling_features(df: pd.DataFrame, windows: list) -> pd.DataFrame:
    """
    Add rolling window statistics on SOH and discharge capacity.
    Groups by cell_id to avoid leakage across cells.
    """
    df = df.sort_values(["cell_id", "cycle_index"]).copy()

    for w in windows:
        grp = df.groupby("cell_id")

        df[f"soh_roll_mean_{w}"] = grp["soh"].transform(
            lambda x: x.rolling(w, min_periods=1).mean()
        )
        df[f"soh_roll_std_{w}"] = grp["soh"].transform(
            lambda x: x.rolling(w, min_periods=1).std().fillna(0)
        )
        df[f"capacity_roll_mean_{w}"] = grp["discharge_capacity"].transform(
            lambda x: x.rolling(w, min_periods=1).mean()
        )
        # Capacity fade rate: slope of SOH over window (linear regression slope)
        df[f"soh_fade_rate_{w}"] = grp["soh"].transform(
            lambda x: x.rolling(w, min_periods=2).apply(
                lambda s: np.polyfit(range(len(s)), s, 1)[0] if len(s) > 1 else 0,
                raw=True
            ).fillna(0)
        )

    return df


def add_lag_features(df: pd.DataFrame, lags: list = [1, 2, 5]) -> pd.DataFrame:
    """
    Add lagged SOH values. These give the model direct access to
    recent history without relying solely on rolling stats.
    """
    df = df.sort_values(["cell_id", "cycle_index"]).copy()
    grp = df.groupby("cell_id")

    for lag in lags:
        df[f"soh_lag_{lag}"] = grp["soh"].transform(lambda x: x.shift(lag))
        df[f"capacity_lag_{lag}"] = grp["discharge_capacity"].transform(lambda x: x.shift(lag))

    return df


def add_cumulative_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features that capture the cell's cumulative history:
    - Cumulative energy throughput (proxy for total stress)
    - Cycle number normalized by max cycles (relative aging position)
    """
    df = df.copy()
    grp = df.groupby("cell_id")

    df["cumulative_capacity"] = grp["discharge_capacity"].transform("cumsum")
    df["cycle_norm"] = df.groupby("cell_id")["cycle_index"].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
    )

    return df


def build_feature_matrix(cycle_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Entry point for feature engineering on cycle-level data.
    ICA/DVA features should be precomputed and joined separately
    (they require raw time-series — see calce_loader.py).

    This function handles all cycle-level statistical features.
    """
    windows = config["features"]["rolling_windows"]

    df = cycle_df.copy()
    df = add_rolling_features(df, windows)
    df = add_lag_features(df)
    df = add_cumulative_features(df)

    # Drop rows with NaN RUL (cells that didn't reach EOL)
    before = len(df)
    df = df.dropna(subset=["rul"])
    after = len(df)
    if before != after:
        logger.info(f"Dropped {before - after} rows with NaN RUL (cells not yet at EOL)")

    # Drop early cycles where lags can't be computed
    df = df.dropna(subset=[c for c in df.columns if "lag" in c])

    logger.info(f"Feature matrix: {df.shape[0]} rows × {df.shape[1]} columns")
    return df.reset_index(drop=True)


def run(config_path: str = "configs/base.yaml"):
    config = load_config(config_path)
    processed_dir = Path(config["data"]["processed_dir"])
    features_dir = Path(config["data"]["features_dir"])
    features_dir.mkdir(parents=True, exist_ok=True)

    for series in config["data"]["calce_series"]:
        input_path = processed_dir / f"calce_{series.lower()}_cycles.parquet"
        if not input_path.exists():
            logger.warning(f"Processed file not found: {input_path} — run ingestion first")
            continue

        logger.info(f"Building features for {series}")
        df = pd.read_parquet(input_path)
        feature_df = build_feature_matrix(df, config)

        output_path = features_dir / f"calce_{series.lower()}_features.parquet"
        feature_df.to_parquet(output_path, index=False)
        logger.info(f"Saved: {output_path}")


if __name__ == "__main__":
    run()

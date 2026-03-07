"""
src/ingestion/calce_loader.py

Ingests raw CALCE battery data from Excel files and extracts
per-cycle summary statistics and SOH values.

CALCE data structure:
- One folder per cell (e.g., CS2_33, CS2_34, CS2_35, CS2_36)
- Multiple Excel files per cell (date-stamped)
- Each file contains time-series columns:
    Cycle_Index, Step_Index, Step_Time, Current(A),
    Voltage(V), Charge_Capacity(Ah), Discharge_Capacity(Ah), Temperature (C)
"""

import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/base.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_cell_files(cell_dir: Path) -> pd.DataFrame:
    """
    Load and concatenate all Excel files for a single cell.
    CALCE splits one cell's data across multiple date-stamped files.
    """
    files = sorted(cell_dir.glob("*.xlsx")) + sorted(cell_dir.glob("*.xls"))

    if not files:
        logger.warning(f"No Excel files found in {cell_dir}")
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            df = pd.read_excel(f, engine="openpyxl")
            dfs.append(df)
        except Exception as e:
            logger.error(f"Failed to read {f}: {e}")

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values("Cycle_Index").reset_index(drop=True)
    return combined


def extract_cycles(raw_df: pd.DataFrame, nominal_capacity: float, eol_threshold: float) -> pd.DataFrame:
    """
    Aggregate raw time-series into per-cycle summary.

    Each row in output = one discharge cycle with:
    - cycle_index
    - discharge_capacity (Ah)
    - SOH (State of Health, 0–1)
    - RUL (Remaining Useful Life in cycles, computed after EOL is known)
    - mean/min/max voltage, temperature during discharge
    - charge_time (cycles to reach full charge)

    SOH = Q_discharge_cycle_n / Q_nominal
    EOL = first cycle where SOH < eol_threshold
    RUL = EOL_cycle - current_cycle  (only valid up to EOL)
    """
    # Identify discharge cycles (current negative in CALCE convention)
    discharge = raw_df[raw_df["Current(A)"] < 0].copy()

    cycle_stats = (
        discharge.groupby("Cycle_Index")
        .agg(
            discharge_capacity=("Discharge_Capacity(Ah)", "max"),
            voltage_mean=("Voltage(V)", "mean"),
            voltage_min=("Voltage(V)", "min"),
            voltage_max=("Voltage(V)", "max"),
            temp_mean=("Temperature (C)", "mean"),
            temp_max=("Temperature (C)", "max"),
        )
        .reset_index()
        .rename(columns={"Cycle_Index": "cycle_index"})
    )

    # Remove clearly bad cycles (capacity = 0 or implausibly large)
    cycle_stats = cycle_stats[
        (cycle_stats["discharge_capacity"] > 0.1) &
        (cycle_stats["discharge_capacity"] < nominal_capacity * 1.1)
    ].copy()

    # SOH
    cycle_stats["soh"] = cycle_stats["discharge_capacity"] / nominal_capacity
    cycle_stats["soh"] = cycle_stats["soh"].clip(0, 1)

    # EOL and RUL
    eol_mask = cycle_stats["soh"] < eol_threshold
    if eol_mask.any():
        eol_cycle = cycle_stats.loc[eol_mask, "cycle_index"].iloc[0]
        cycle_stats = cycle_stats[cycle_stats["cycle_index"] <= eol_cycle].copy()
        cycle_stats["rul"] = eol_cycle - cycle_stats["cycle_index"]
    else:
        # Cell hasn't reached EOL — RUL is unknown, mark as NaN
        logger.warning("Cell has not reached EOL. RUL values will be NaN.")
        cycle_stats["rul"] = np.nan

    cycle_stats = cycle_stats.reset_index(drop=True)
    return cycle_stats


def process_calce_series(series_dir: Path, nominal_capacity: float, config: dict) -> pd.DataFrame:
    """
    Process all cells in a CALCE series directory (e.g., CS2/).
    Returns a combined DataFrame with a 'cell_id' column.
    """
    eol_threshold = config["data"]["eol_threshold"]
    min_cycles = config["data"]["min_cycles"]

    all_cells = []
    cell_dirs = [d for d in series_dir.iterdir() if d.is_dir()]

    if not cell_dirs:
        logger.error(f"No cell directories found in {series_dir}")
        return pd.DataFrame()

    for cell_dir in sorted(cell_dirs):
        cell_id = cell_dir.name
        logger.info(f"Processing cell: {cell_id}")

        raw_df = load_cell_files(cell_dir)
        if raw_df.empty:
            continue

        cycle_df = extract_cycles(raw_df, nominal_capacity, eol_threshold)

        if len(cycle_df) < min_cycles:
            logger.warning(f"Cell {cell_id} has only {len(cycle_df)} cycles — skipping (min={min_cycles})")
            continue

        cycle_df.insert(0, "cell_id", cell_id)
        all_cells.append(cycle_df)
        logger.info(f"  → {len(cycle_df)} cycles extracted, EOL: {cycle_df['rul'].min() == 0}")

    if not all_cells:
        return pd.DataFrame()

    return pd.concat(all_cells, ignore_index=True)


def run(config_path: str = "configs/base.yaml"):
    config = load_config(config_path)
    raw_dir = Path(config["data"]["raw_dir"])
    processed_dir = Path(config["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    # CALCE nominal capacities per series
    nominal_capacities = {
        "CS2": 1.1,   # Ah
        "CX2": 1.35,  # Ah
    }

    for series in config["data"]["calce_series"]:
        series_dir = raw_dir / "calce" / series
        if not series_dir.exists():
            logger.warning(f"Series directory not found: {series_dir} — skipping")
            continue

        logger.info(f"Processing CALCE series: {series}")
        df = process_calce_series(series_dir, nominal_capacities[series], config)

        if df.empty:
            logger.warning(f"No data extracted for series {series}")
            continue

        output_path = processed_dir / f"calce_{series.lower()}_cycles.parquet"
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved: {output_path} ({len(df)} rows, {df['cell_id'].nunique()} cells)")


if __name__ == "__main__":
    run()

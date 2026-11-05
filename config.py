"""
config.py - every tunable setting for the detection pipeline lives here.

Override anything without editing code:
    from config import Config
    cfg = Config(contamination=0.01, z_threshold=3.0)
or via command line flags in main.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent

# Column that holds the zero-based row position of every record in the ORIGINAL csv.
# (csv line number in a text editor = source_row_id + 2, because of the header line.)
SOURCE_ROW_ID = "source_row_id"


@dataclass
class Config:
    # ------------------------------------------------------------------ paths
    input_csv: Path = PROJECT_ROOT / "data" / "EmissioNex_processed_dataset.csv"
    output_dir: Path = PROJECT_ROOT / "outputs"

    # ---------------------------------------------------------------- columns
    timestamp_col: str = "timestamp"
    target_col: str = "kWh"                      # quantity the baseline predicts
    expected_col: str = "expected_kWh"           # name used for the baseline prediction
    actual_co2e_col: str = "co2e_electricity"    # dataset's own kg CO2e column (optional)
    label_flag_col: str = "fault_flag"           # injected-fault label (EVALUATION ONLY)
    label_type_col: str = "fault_type"           # injected-fault label (EVALUATION ONLY)
    # facility hierarchy, top -> bottom. Missing levels are skipped automatically.
    hierarchy: List[str] = field(default_factory=lambda: [
        "plant_id", "building_id", "department_id", "machine_id"])
    machine_col: str = "machine_id"

    # --------------------------------------------------------- baseline model
    categorical_features: List[str] = field(default_factory=lambda: [
        "plant_id", "building_id", "department_id", "machine_id",
        "machine_type", "operating_status"])
    # Same candidate list as the Member 2 notebook. Columns that fail the leakage
    # check (see preprocessing.check_rolling_leakage) are removed automatically.
    numeric_features: List[str] = field(default_factory=lambda: [
        "hour", "day_of_week", "is_weekend", "is_off_shift",
        "power_factor", "production_output",
        "previous_hour_kwh", "rolling_3h_kwh", "rolling_6h_kwh", "rolling_24h_kwh"])
    # Columns that are only allowed if verified as "past only" w.r.t. the target.
    leak_suspect_columns: Dict[str, int] = field(default_factory=lambda: {
        # column: window length in 15-minute intervals (sum of kWh)
        "previous_hour_kwh": 4,
        "rolling_3h_kwh": 12,
        "rolling_6h_kwh": 24,
        "rolling_24h_kwh": 96})
    baseline_strategy: str = "blocked_oof"       # blocked_oof | chronological_holdout | median
    n_splits: int = 5                            # time blocks for blocked_oof
    train_fraction: float = 0.80                 # chronological_holdout only
    min_rows_for_model: int = 500                # below this -> median fallback
    exclude_labeled_faults_from_training: bool = False   # labels are not available in production
    fit_final_model: bool = True                 # extra model on all data for future inference
    xgb_params: Dict = field(default_factory=lambda: dict(
        n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, objective="reg:squarederror", random_state=42, n_jobs=1))
    median_min_group_size: int = 5
    random_state: int = 42

    # ----------------------------------------------------- deviation settings
    emission_factor_kg_per_kwh: float = 0.82     # grid factor from the project brief
    pct_min_expected_kwh: float = 1.0            # % deviation undefined below this expected value

    # ------------------------------------------------------ anomaly detection
    z_threshold: float = 2.5                     # residual z-score threshold (sigma)
    z_medium: float = 3.5                        # severity bands on residual z-score
    z_high: float = 5.0
    contamination: float = 0.02                  # Isolation Forest anomaly share
    n_estimators_if: int = 200
    if_features: List[str] = field(default_factory=lambda: [
        "residual_z", "residual_z_roll4", "residual_z_roll12", "residual_kWh"])
    roll_short: int = 4                          # trailing intervals (= 1 h at 15 min)
    roll_long: int = 12                          # trailing intervals (= 3 h at 15 min)
    combine_mode: str = "intersection"           # intersection | union | if_only | z_only
    direction: str = "increase"                  # increase | both
    min_rows_for_if: int = 50                    # below this Isolation Forest is skipped
    min_group_size_for_z: int = 30               # per-machine z needs >= this many residuals

    # ------------------------------------------------------------ localization
    event_gap_intervals: int = 4                 # anomalies <= this many rows apart = one event
    daily_freq: str = "1D"
    window_freq: str = "6h"                      # sustained-deviation window
    window_z_threshold: float = 2.5
    window_min_fraction: float = 0.5             # window needs >= 50% of its expected rows
    window_min_windows_per_machine: int = 8

    def to_dict(self) -> dict:
        d = asdict(self)
        d["input_csv"] = str(self.input_csv)
        d["output_dir"] = str(self.output_dir)
        return d

    def with_overrides(self, **kw) -> "Config":
        return replace(self, **kw)


DEFAULT_CONFIG = Config()

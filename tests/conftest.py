import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import Config  # noqa: E402


def make_synthetic(n_days=10, spike=True, seed=0):
    """4 machines (2 plants/2 buildings), 15-min data, deterministic daily pattern + noise.
    Machine M3 gets a 3-hour +60 kWh excess on day 7 (labelled with fault_flag)."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-02-01", periods=n_days * 96, freq="15min")
    rows = []
    layout = {"M1": ("P1", "B1", "D1"), "M2": ("P1", "B1", "D2"),
              "M3": ("P2", "B2", "D3"), "M4": ("P2", "B2", "D4")}
    for mid, (p, b, d) in layout.items():
        hour = ts.hour.to_numpy()
        running = ((hour >= 8) & (hour < 18)).astype(int)
        base = np.where(running == 1, 80.0, 10.0)
        kwh = base + rng.normal(0, 3, len(ts))
        flag = np.zeros(len(ts), int)
        if spike and mid == "M3":
            s = np.where((ts.day == 8) & (hour >= 10) & (hour < 13))[0]
            kwh[s] += 60.0
            flag[s] = 1
        rows.append(pd.DataFrame({
            "timestamp": ts, "plant_id": p, "building_id": b, "department_id": d, "machine_id": mid,
            "machine_type": "Motor", "operating_status": np.where(running == 1, "Running", "Idle"),
            "kWh": kwh.clip(min=0.1), "power_factor": 0.95, "production_output": running * 100.0,
            "hour": hour, "day_of_week": ts.dayofweek, "is_weekend": (ts.dayofweek >= 5).astype(int),
            "is_off_shift": 1 - running, "fault_flag": flag,
            "fault_type": np.where(flag == 1, "Test_fault", None)}))
    df = pd.concat(rows, ignore_index=True)
    df["co2e_electricity"] = df["kWh"] * 0.82
    return df.sample(frac=1.0, random_state=1).reset_index(drop=True)   # shuffled on purpose


@pytest.fixture(scope="session")
def synthetic():
    return make_synthetic()


@pytest.fixture()
def fast_cfg(tmp_path):
    return Config(output_dir=tmp_path,
                  xgb_params=dict(n_estimators=40, max_depth=4, learning_rate=0.2, subsample=0.8,
                                  colsample_bytree=0.8, objective="reg:squarederror",
                                  random_state=42, n_jobs=1),
                  n_splits=4, min_rows_for_model=200,
                  numeric_features=["hour", "day_of_week", "is_weekend", "is_off_shift",
                                    "power_factor", "production_output"],
                  leak_suspect_columns={})

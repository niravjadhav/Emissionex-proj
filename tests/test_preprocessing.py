import numpy as np
import pandas as pd
import pytest

from config import Config, SOURCE_ROW_ID
from preprocessing import (SchemaError, attach_source_row_id, check_rolling_leakage, clean_dataset,
                           select_baseline_features, validate_schema)
from tests.conftest import make_synthetic


def test_missing_required_column_raises():
    df = make_synthetic(2).drop(columns=["kWh"])
    with pytest.raises(SchemaError):
        validate_schema(df, Config())


def test_empty_dataset_raises():
    with pytest.raises(SchemaError):
        validate_schema(make_synthetic(2).iloc[0:0], Config())


def test_source_row_id_preserved_through_cleaning():
    raw = attach_source_row_id(make_synthetic(2))
    clean, rep = clean_dataset(raw, Config())
    assert clean[SOURCE_ROW_ID].is_unique
    merged = clean.merge(raw[[SOURCE_ROW_ID, "kWh"]], on=SOURCE_ROW_ID, suffixes=("", "_raw"))
    assert np.allclose(merged["kWh"], merged["kWh_raw"])


def test_duplicates_bad_timestamps_negative_values():
    raw = attach_source_row_id(make_synthetic(2))
    raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)       # exact duplicate
    raw["timestamp"] = raw["timestamp"].astype(str)
    raw.loc[1, "timestamp"] = "not-a-date"
    raw.loc[2, "kWh"] = -5.0
    clean, rep = clean_dataset(raw, Config())
    assert rep["dropped_invalid_timestamp"] == 1
    assert rep["negative_target_set_to_nan"] == 1
    assert rep["dropped_exact_duplicates"] + rep["dropped_duplicate_timestamp_machine"] >= 1
    assert clean["kWh"].isna().sum() >= 1                          # kept, not scored


def test_leakage_check_detects_current_interval_rolling_sum():
    df = make_synthetic(4)
    df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)
    g = df.groupby("machine_id")["kWh"]
    df["leaky"] = g.transform(lambda s: s.rolling(12, min_periods=1).sum())
    df["safe"] = g.transform(lambda s: s.rolling(4, min_periods=1).sum().shift(1))
    cfg = Config(leak_suspect_columns={"leaky": 12, "safe": 4}, numeric_features=["hour", "leaky", "safe"])
    rep = check_rolling_leakage(df, cfg)
    assert rep["leaky"]["verdict"] == "includes_current_interval"
    assert rep["safe"]["verdict"] == "past_only"
    cats, nums, excl = select_baseline_features(df, cfg, rep)
    assert "leaky" in excl and "leaky" not in nums
    assert "safe" in nums

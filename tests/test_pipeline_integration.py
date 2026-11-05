"""End-to-end tests: CSV -> baseline -> detection -> localization -> files."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from baseline import encode_features, load_baseline_artifacts
from config import Config, SOURCE_ROW_ID
from main import main
from pipeline import run_detection
from reporting import write_outputs
from tests.conftest import make_synthetic

REAL = Path(__file__).resolve().parents[1] / "data" / "EmissioNex_processed_dataset.csv"

REQUIRED_RESULT_COLS = [SOURCE_ROW_ID, "timestamp", "plant_id", "building_id", "department_id",
                        "machine_id", "kWh", "expected_kWh", "residual_kWh", "abs_deviation_kWh",
                        "pct_deviation", "excess_co2e_kg", "residual_z", "anomaly_score",
                        "is_anomaly", "severity", "detection_status", "baseline_model_id"]


def test_end_to_end_synthetic_finds_injected_fault_and_writes_files(synthetic, fast_cfg, tmp_path):
    out = run_detection(synthetic, fast_cfg)
    assert all(c in out.results.columns for c in REQUIRED_RESULT_COLS)
    assert len(out.results) == len(synthetic)
    ev = out.evaluation
    assert ev["available"] and ev["event_recall"] == 1.0
    assert out.drilldown[out.drilldown["level"] == "machine"]["node"].iloc[0] == "M3"
    paths = write_outputs(out, tmp_path)
    for k in ("detection_results", "detected_incidents", "localization_summary", "evaluation_report"):
        assert Path(paths[k]).exists() and Path(paths[k]).stat().st_size > 0
    rep = Path(paths["evaluation_report"]).read_text()
    assert "LIMITATIONS" in rep and "STATISTICAL anomaly" in rep


def test_member4_can_map_back_and_rebuild_shap_inputs(synthetic, fast_cfg, tmp_path):
    out = run_detection(synthetic, fast_cfg)
    write_outputs(out, tmp_path)
    inc = pd.read_csv(tmp_path / "detected_incidents.csv")
    assert len(inc) > 0
    # 1) source_row_id maps to the ORIGINAL input rows and values are unchanged
    orig = synthetic.loc[inc[SOURCE_ROW_ID].values]
    assert np.allclose(orig["kWh"].values, inc["kWh"].values)
    assert (orig["machine_id"].values == inc["machine_id"].values).all()
    # 2) saved models + columns rebuild the exact encoded matrix and reproduce expected_kWh
    art = load_baseline_artifacts(tmp_path / "baseline_models")
    for mid, grp in inc.groupby("baseline_model_id"):
        X = encode_features(grp, art.categorical, art.numeric, art.feature_columns)
        pred = art.models[mid].predict(X)
        assert np.allclose(pred, grp["expected_kWh"].values, atol=1e-3)
    # 3) ground-truth columns are clearly renamed
    assert "label_fault_flag" in inc.columns and "fault_flag" not in inc.columns


def test_external_expected_values_are_respected(synthetic, fast_cfg):
    ext = np.full(len(synthetic), 50.0)
    out = run_detection(synthetic, fast_cfg, expected_kwh=ext)
    assert out.baseline is None
    assert np.allclose(out.results["expected_kWh"], 50.0)
    assert (out.results["baseline_model_id"] == "external").all()


def test_small_dataset_falls_back_to_median_baseline(fast_cfg):
    small = make_synthetic(n_days=1).iloc[:150]
    cfg = fast_cfg.with_overrides(min_rows_for_model=500)
    out = run_detection(small, cfg)
    assert out.baseline.method.startswith("Robust median")
    assert any("too small" in w for w in out.warnings)
    assert len(out.results) == len(small)


def test_no_anomalies_scenario_still_writes_valid_outputs(synthetic, fast_cfg, tmp_path):
    cfg = fast_cfg.with_overrides(z_threshold=100.0)
    out = run_detection(synthetic, cfg)
    assert len(out.incidents) == 0
    paths = write_outputs(out, tmp_path)
    inc = pd.read_csv(paths["detected_incidents"])
    assert len(inc) == 0 and "incident_id" in inc.columns
    assert "No anomalies detected" in " ".join(out.warnings)


def test_missing_target_values_do_not_crash(synthetic, fast_cfg):
    d = synthetic.copy()
    d.loc[d.index[:25], "kWh"] = np.nan
    out = run_detection(d, fast_cfg)
    assert (out.results["detection_status"] == "not_scored").sum() >= 25


def test_chronological_holdout_scores_only_later_rows(synthetic, fast_cfg):
    out = run_detection(synthetic, fast_cfg.with_overrides(baseline_strategy="chronological_holdout"))
    ns = out.results["detection_status"] == "not_scored"
    assert ns.any() and (~ns).any()
    assert out.results.loc[~ns, "timestamp"].min() > out.results.loc[ns, "timestamp"].max()


@pytest.mark.skipif(not REAL.exists(), reason="real dataset not present")
def test_real_dataset_subset_runs_and_leaky_columns_are_excluded(tmp_path):
    raw = pd.read_csv(REAL, low_memory=False)
    ts = sorted(raw["timestamp"].unique())
    sub = raw[raw["timestamp"].isin(ts[: 96 * 14])]            # first ~14 days, all machines
    cfg = Config(output_dir=tmp_path, n_splits=3,
                 xgb_params=dict(n_estimators=40, max_depth=4, learning_rate=0.2,
                                 objective="reg:squarederror", random_state=42, n_jobs=1))
    out = run_detection(sub, cfg)
    assert set(out.excluded_features) == {"rolling_3h_kwh", "rolling_6h_kwh", "rolling_24h_kwh"}
    assert "previous_hour_kwh" in out.baseline.numeric
    assert out.leakage_report["previous_hour_kwh"]["verdict"] == "past_only"
    assert len(out.results) == len(sub)


def test_cli_main_runs_end_to_end(tmp_path, synthetic, monkeypatch):
    csv = tmp_path / "in.csv"
    synthetic.to_csv(csv, index=False)
    outdir = tmp_path / "out"
    import config as cfgmod
    fast = dict(n_estimators=30, max_depth=3, learning_rate=0.2, objective="reg:squarederror",
                random_state=42, n_jobs=1)
    monkeypatch.setattr(cfgmod.DEFAULT_CONFIG, "xgb_params", fast)
    monkeypatch.setattr(cfgmod.DEFAULT_CONFIG, "numeric_features",
                        ["hour", "day_of_week", "is_weekend", "is_off_shift", "power_factor", "production_output"])
    monkeypatch.setattr(cfgmod.DEFAULT_CONFIG, "leak_suspect_columns", {})
    rc = main(["--input", str(csv), "--output-dir", str(outdir), "--no-save-models"])
    assert rc == 0 and (outdir / "evaluation_report.txt").exists()
    assert main(["--input", str(tmp_path / "missing.csv")]) == 2
    assert main(["--input", str(csv), "--contamination", "0.9"]) == 2

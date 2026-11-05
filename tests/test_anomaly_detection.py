import numpy as np
import pandas as pd

from anomaly_detection import AnomalyDetector, apply_zscore, compute_deviation, robust_stats
from config import Config


def _dev(n=600, seed=0, spike_idx=()):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-01-01", periods=n, freq="15min")
    exp = pd.Series(50.0 + rng.normal(0, 0.1, n))
    act = exp + rng.normal(0, 2.0, n)
    for i in spike_idx:
        act[i] += 25
    df = pd.DataFrame({"timestamp": ts, "machine_id": "M1", "kWh": act})
    return compute_deviation(df, exp, Config())


def test_deviation_maths_and_pct_validity():
    cfg = Config()
    df = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=3, freq="15min"),
                       "machine_id": "M1", "kWh": [12.0, 5.0, 3.0]})
    out = compute_deviation(df, pd.Series([10.0, 0.0, 0.5]), cfg)
    assert np.allclose(out["residual_kWh"], [2.0, 5.0, 2.5])
    assert out["pct_deviation"].iloc[0] == 20.0
    assert np.isnan(out["pct_deviation"].iloc[1])            # expected == 0 -> undefined
    assert np.isnan(out["pct_deviation"].iloc[2])            # expected below minimum
    assert np.allclose(out["excess_co2e_kg"], out["residual_kWh"] * 0.82)


def test_robust_z_constant_series_has_no_division_error():
    s = pd.Series([1.0] * 50)
    stats = robust_stats(s, pd.Series(["a"] * 50), 5)
    z = apply_zscore(s, pd.Series(["a"] * 50), stats)
    assert not np.isinf(z.dropna()).any()


def test_detector_finds_obvious_spikes_and_labels_status():
    dev = _dev(spike_idx=range(300, 306))
    cfg = Config(contamination=0.02)
    res = AnomalyDetector(cfg).fit(dev).score(dev)
    flagged = set(np.where(res["is_anomaly"])[0])
    assert len(flagged & set(range(300, 306))) >= 5
    assert set(res["detection_status"].unique()) <= {"anomaly", "normal", "not_scored"}
    assert (res.loc[res["is_anomaly"], "direction"] == "increase").all()
    assert (res.loc[res["is_anomaly"], "severity"] != "none").all()
    assert (res.loc[~res["is_anomaly"], "severity"] == "none").all()


def test_contamination_is_configurable():
    dev = _dev(spike_idx=range(300, 306))
    n_lo = AnomalyDetector(Config(contamination=0.005, combine_mode="if_only", direction="both")).fit(dev).score(dev)["is_anomaly"].sum()
    n_hi = AnomalyDetector(Config(contamination=0.10, combine_mode="if_only", direction="both")).fit(dev).score(dev)["is_anomaly"].sum()
    assert n_hi > n_lo


def test_no_anomalies_when_threshold_extreme():
    dev = _dev()
    res = AnomalyDetector(Config(z_threshold=50.0)).fit(dev).score(dev)
    assert res["is_anomaly"].sum() == 0


def test_missing_values_are_not_scored_not_crashing():
    dev = _dev(spike_idx=[300])
    dev.loc[10:20, "residual_kWh"] = np.nan
    res = AnomalyDetector(Config()).fit(dev).score(dev)
    assert (res.loc[10:20, "detection_status"] == "not_scored").all()
    assert not res.loc[10:20, "is_anomaly"].any()


def test_small_dataset_skips_isolation_forest_but_still_runs():
    dev = _dev(n=40, spike_idx=[30])
    det = AnomalyDetector(Config(min_group_size_for_z=10)).fit(dev)
    assert det.iso_ is None and det.warnings_
    res = det.score(dev)
    assert "is_anomaly" in res and res["if_flag"].sum() == 0

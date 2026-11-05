import numpy as np
import pandas as pd

from anomaly_detection import AnomalyDetector, compute_deviation
from config import Config
from localization import (assign_events, build_incidents, detect_window_anomalies, drill_down,
                          summarize_locations)


def _results(n_days=12):
    rng = np.random.default_rng(3)
    ts = pd.date_range("2026-01-01", periods=n_days * 96, freq="15min")
    frames = []
    for mid, (p, b, d) in {"M1": ("P1", "B1", "D1"), "M2": ("P1", "B1", "D2"), "M3": ("P2", "B2", "D3")}.items():
        exp = pd.Series(40.0, index=range(len(ts)))
        act = exp + rng.normal(0, 2, len(ts))
        if mid == "M2":
            sl = slice(500, 524)                       # 6 h sustained excess
            act[sl] += 15
        frames.append(pd.DataFrame({"timestamp": ts, "plant_id": p, "building_id": b,
                                    "department_id": d, "machine_id": mid, "kWh": act,
                                    "expected": exp}))
    df = pd.concat(frames, ignore_index=True)
    cfg = Config(min_group_size_for_z=10)
    dev = compute_deviation(df.drop(columns="expected"), df["expected"], cfg)
    res = AnomalyDetector(cfg).fit(dev).score(dev)
    return res, cfg


def test_events_incidents_summary_and_ranking():
    res, cfg = _results()
    res = assign_events(res, cfg)
    inc = build_incidents(res, cfg)
    assert len(inc) == res["is_anomaly"].sum() > 0
    assert inc["incident_id"].is_unique
    assert (res.loc[res["is_anomaly"], "event_id"] != "").all()
    assert (res.loc[~res["is_anomaly"], "event_id"] == "").all()
    s = summarize_locations(res, cfg)
    m = s[(s["level"] == "machine") & (s["time_window"] == "ALL")].sort_values("rank_in_level")
    assert m.iloc[0]["machine_id"] == "M2"
    assert set(["plant", "building", "department", "machine"]) <= set(s["level"])
    # ancestors preserved on every machine row
    assert m["building_id"].notna().all() and m["plant_id"].notna().all()


def test_drill_down_traces_to_faulty_machine():
    res, cfg = _results()
    path = drill_down(res, cfg)
    assert list(path["level"])[:4] == ["plant", "building", "department", "machine"]
    assert path[path["level"] == "machine"]["node"].iloc[0] == "M2"


def test_window_detection_flags_sustained_window_and_ignores_sparse_windows():
    res, cfg = _results()
    w = detect_window_anomalies(res, cfg)
    assert len(w) and (w["machine_id"] == "M2").any()
    # a window with < 50% of expected rows must never be flagged
    sparse = res[res["machine_id"] == "M2"].copy()
    sparse = sparse[~((sparse["timestamp"].dt.hour.between(0, 5)) & (sparse["timestamp"].dt.hour != 0))]
    w2 = detect_window_anomalies(sparse, cfg)
    assert w2.empty or (w2["n_records_scored"] >= 12).all()


def test_localization_handles_missing_hierarchy_columns():
    res, cfg = _results()
    res = res.drop(columns=["building_id", "department_id"])
    res = assign_events(res, cfg)
    s = summarize_locations(res, cfg)
    assert set(s["level"]) == {"plant", "machine"}


def test_no_anomalies_gives_empty_incidents_and_valid_summary():
    res, cfg = _results()
    res["is_anomaly"] = False
    res["severity"] = "none"
    res = assign_events(res, cfg)
    inc = build_incidents(res, cfg)
    assert len(inc) == 0 and "incident_id" in inc.columns
    assert len(drill_down(res, cfg)) == 0
    s = summarize_locations(res, cfg)
    assert (s["n_anomalies"] == 0).all()

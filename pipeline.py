"""
pipeline.py - the single entry point for the detection and explainability workflows.

    from pipeline import run_detection
    out = run_detection("data/EmissioNex_processed_dataset.csv")      # path or DataFrame
    incidents = out.incidents                                           # detected anomalies

Optional: supply your own expected-emission baseline (e.g. Member 2's model) by passing
`expected_kwh=` (Series/array aligned with the input rows) or by including an
`expected_kWh` column in the DataFrame.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from anomaly_detection import AnomalyDetector, compute_deviation
from baseline import BaselineResult, fit_expected_baseline, median_baseline
from config import Config, DEFAULT_CONFIG, SOURCE_ROW_ID
from evaluation import evaluate_against_labels
from localization import (assign_events, build_incidents, detect_window_anomalies,
                          drill_down, summarize_locations)
from preprocessing import (attach_source_row_id, check_rolling_leakage, clean_dataset,
                           load_dataset, select_baseline_features, validate_schema)


@dataclass
class DetectionOutput:
    results: pd.DataFrame                  # every record + flags (detection_results.csv)
    incidents: pd.DataFrame                # anomalies only (detected_incidents.csv)
    summary: pd.DataFrame                  # localization_summary.csv (levels + windows)
    window_anomalies: pd.DataFrame
    drilldown: pd.DataFrame
    baseline: Optional[BaselineResult]
    schema_report: Dict
    clean_report: Dict
    leakage_report: Dict
    excluded_features: Dict
    evaluation: Dict
    config: Config
    warnings: List[str] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)


def run_detection(data: Union[str, pd.DataFrame, None] = None, cfg: Optional[Config] = None,
                  expected_kwh=None) -> DetectionOutput:
    cfg = cfg or DEFAULT_CONFIG
    t0 = time.perf_counter()
    timings: Dict[str, float] = {}
    warnings: List[str] = []

    # 1. load / validate / clean --------------------------------------------------------
    if data is None:
        data = cfg.input_csv
    raw = load_dataset(data) if not isinstance(data, pd.DataFrame) else attach_source_row_id(data)
    schema = validate_schema(raw, cfg)
    df, clean_report = clean_dataset(raw, cfg)
    if df.empty:
        raise ValueError("No usable rows after cleaning.")
    timings["load_clean_s"] = time.perf_counter() - t0

    # 2. expected-emission baseline ------------------------------------------------------
    t1 = time.perf_counter()
    leak, excluded, baseline = {}, {}, None
    if expected_kwh is not None:
        exp = _align_external_expected(expected_kwh, raw, df)
        method_note = "externally supplied expected_kwh"
    elif cfg.expected_col in df.columns and df[cfg.expected_col].notna().any():
        exp = pd.to_numeric(df[cfg.expected_col], errors="coerce")
        method_note = f"existing '{cfg.expected_col}' column in the input"
    else:
        leak = check_rolling_leakage(df, cfg)
        cats, nums, excluded = select_baseline_features(df, cfg, leak)
        baseline = fit_expected_baseline(df, cfg, cats, nums)
        exp = baseline.expected
        method_note = baseline.method
        warnings.extend(baseline.notes)
    if exp.notna().sum() == 0:
        raise ValueError("No expected-emission values could be produced.")
    timings["baseline_s"] = time.perf_counter() - t1

    # 3. detection -------------------------------------------------------------------------
    t2 = time.perf_counter()
    dev = compute_deviation(df, exp, cfg)
    det = AnomalyDetector(cfg).fit(dev)
    res = det.score(dev)
    warnings.extend(det.warnings_)
    if baseline is not None:
        res["baseline_model_id"] = baseline.model_id.reindex(res.index).values
    else:
        res["baseline_model_id"] = "external"
    res["baseline_method"] = method_note
    timings["detection_s"] = time.perf_counter() - t2

    # 4. localization ------------------------------------------------------------------------
    t3 = time.perf_counter()
    res = assign_events(res, cfg)
    res = _rename_labels(res, cfg)
    res = _order_columns(res, cfg, baseline)
    incidents = build_incidents(res, cfg)
    summary_levels = summarize_locations(res, cfg)
    win = detect_window_anomalies(res, cfg)
    summary = _merge_windows(summary_levels, win, cfg)
    dd = drill_down(res, cfg)
    timings["localization_s"] = time.perf_counter() - t3

    ev = evaluate_against_labels(_restore_label_names(res, cfg), cfg, win, summary)
    timings["total_s"] = time.perf_counter() - t0
    if len(incidents) == 0:
        warnings.append("No anomalies detected with the current configuration.")
    return DetectionOutput(res, incidents, summary, win, dd, baseline, schema, clean_report, leak,
                           excluded, ev, cfg, warnings, timings)


# ------------------------------------------------------------------------- helpers
def _align_external_expected(expected, raw: pd.DataFrame, clean: pd.DataFrame) -> pd.Series:
    """External predictions must line up with the ORIGINAL input rows (raw order)."""
    arr = pd.Series(np.asarray(expected, dtype=float).ravel())
    if len(arr) != len(raw):
        raise ValueError(f"expected_kwh has {len(arr)} values but input has {len(raw)} rows.")
    arr.index = raw[SOURCE_ROW_ID].values
    return pd.Series(arr.reindex(clean[SOURCE_ROW_ID].values).values, index=clean.index)


def _rename_labels(res: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Rename ground-truth columns so nobody feeds them to SHAP by accident."""
    ren = {c: f"label_{c}" for c in (cfg.label_flag_col, cfg.label_type_col) if c in res.columns}
    return res.rename(columns=ren)


def _restore_label_names(res: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    ren = {f"label_{c}": c for c in (cfg.label_flag_col, cfg.label_type_col)
           if f"label_{c}" in res.columns}
    return res.rename(columns=ren)


def _order_columns(res: pd.DataFrame, cfg: Config, baseline: Optional[BaselineResult]) -> pd.DataFrame:
    ids = [SOURCE_ROW_ID, cfg.timestamp_col] + [c for c in cfg.hierarchy if c in res.columns] + \
          [c for c in ("machine_type", "operating_status") if c in res.columns]
    core = [cfg.target_col, cfg.expected_col, "residual_kWh", "abs_deviation_kWh", "pct_deviation",
            "actual_co2e_kg", "expected_co2e_kg", "excess_co2e_kg",
            "residual_z", "residual_z_roll4", "residual_z_roll12",
            "anomaly_score", "if_flag", "z_flag", "direction", "is_anomaly", "severity",
            "detection_status", "event_id", "baseline_model_id", "baseline_method"]
    feats = []
    if baseline is not None:
        feats = [c for c in baseline.numeric + baseline.categorical if c not in ids]
    labels = [c for c in res.columns if c.startswith("label_")]
    first = list(dict.fromkeys(ids + core + feats))
    first = [c for c in first if c in res.columns]
    rest = [c for c in res.columns if c not in first and c not in labels]
    return res[first + rest + labels]


def _merge_windows(levels: pd.DataFrame, win: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if win.empty:
        return levels
    w = win.copy()
    w["n_anomalies"] = w["n_row_anomalies"]
    w["total_excess_kwh"] = w["window_excess_kwh"]
    w["total_excess_co2e_kg"] = w["window_excess_co2e_kg"]
    w["rank_in_level"] = w["total_excess_co2e_kg"].rank(ascending=False, method="min")
    keep = [c for c in w.columns if c in levels.columns or c in ("window_z", "window_length", "window_mean_residual_z")]
    w = w[keep]
    return pd.concat([levels, w], ignore_index=True) if len(levels) else w

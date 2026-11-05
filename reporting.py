"""reporting.py - write the four required output files (+ small extras)."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from pipeline import DetectionOutput


def write_outputs(out: DetectionOutput, output_dir=None, save_models: bool = True) -> dict:
    cfg = out.config
    d = Path(output_dir or cfg.output_dir)
    d.mkdir(parents=True, exist_ok=True)
    paths = {
        "detection_results": d / "detection_results.csv",
        "detected_incidents": d / "detected_incidents.csv",
        "localization_summary": d / "localization_summary.csv",
        "drilldown_top_path": d / "drilldown_top_path.csv",
        "evaluation_report": d / "evaluation_report.txt",
    }
    out.results.to_csv(paths["detection_results"], index=False)
    out.incidents.to_csv(paths["detected_incidents"], index=False)
    out.summary.to_csv(paths["localization_summary"], index=False)
    out.drilldown.to_csv(paths["drilldown_top_path"], index=False)
    if out.baseline is not None:
        meta = {"feature_columns": out.baseline.feature_columns, "categorical": out.baseline.categorical,
                "numeric": out.baseline.numeric}
        (d / "baseline_feature_columns.json").write_text(json.dumps(meta, indent=2))
        paths["baseline_feature_columns"] = d / "baseline_feature_columns.json"
        if save_models and out.baseline.models:
            out.baseline.save(d / "baseline_models")
            paths["baseline_models"] = d / "baseline_models"
    paths["evaluation_report"].write_text(build_report(out, paths), encoding="utf-8")
    return {k: str(v) for k, v in paths.items()}


def _f(x, p=4):
    return "n/a" if x is None or (isinstance(x, float) and x != x) else f"{x:.{p}f}"


def build_report(out: DetectionOutput, paths: dict) -> str:
    cfg, r, L = out.config, out.results, []
    A = L.append
    A("=" * 78)
    A("EmissioNex - Anomaly Detection & Localization - evaluation report")
    A(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    A("=" * 78)
    A("\n1. DATA CHECKS")
    sr, cr = out.schema_report, out.clean_report
    A(f"  rows read: {sr['n_rows']}   columns: {sr['n_columns']}   rows after cleaning: {cr['rows_out']}")
    A(f"  time range: {r[cfg.timestamp_col].min()}  ->  {r[cfg.timestamp_col].max()}")
    A(f"  hierarchy levels available: {sr['hierarchy_levels_available']}")
    A(f"  missing optional columns: {sr['missing_optional_columns'] or 'none'}")
    A(f"  null counts (columns with nulls): {sr['null_counts'] or 'none'}")
    A(f"  dropped invalid timestamps: {cr['dropped_invalid_timestamp']}; exact duplicates: "
      f"{cr['dropped_exact_duplicates']}; duplicate (timestamp,machine): {cr['dropped_duplicate_timestamp_machine']}")
    A(f"  negative energy set to NaN: {cr['negative_target_set_to_nan']}; "
      f"rows with missing target (not scored): {cr['missing_target_rows_kept_but_not_scored']}")
    if out.leakage_report:
        A("\n  Target-leakage check on lag/rolling columns (vs per-machine rolling sum of kWh):")
        for c, v in out.leakage_report.items():
            A(f"    {c:18s} window={v['window_intervals']:>3}  match_incl_current={v['match_including_current']:.3f}"
              f"  match_past_only={v['match_past_only']:.3f}  -> {v['verdict']}")
        A(f"  Columns excluded from the baseline: {out.excluded_features or 'none'}")
        if any(v["verdict"] == "includes_current_interval" for v in out.leakage_report.values()):
            A("  NOTE: the Member 2 notebook feeds rolling_3h/6h/24h_kwh to XGBoost. These columns are")
            A("        rolling sums that INCLUDE the current interval's kWh (the prediction target), i.e.")
            A("        target leakage. They are removed here; baseline R2 is therefore a leak-free figure.")

    A("\n2. METHOD")
    A("  Expected-emission baseline: " + (out.baseline.method if out.baseline else r['baseline_method'].iloc[0]))
    if out.baseline:
        A(f"    categorical features: {out.baseline.categorical}")
        A(f"    numeric features    : {out.baseline.numeric}")
        A(f"    encoded columns     : {len(out.baseline.feature_columns)}")
    A("  Deviation: residual = actual kWh - expected kWh; abs deviation = |residual|;")
    A(f"    % deviation = 100*residual/expected only where expected >= {cfg.pct_min_expected_kwh} kWh (else blank);")
    A(f"    excess CO2e (kg) = residual kWh x {cfg.emission_factor_kg_per_kwh} kg/kWh.")
    A("  Residual z-score: robust (median / 1.4826*MAD) per machine, global fallback if < "
      f"{cfg.min_group_size_for_z} residuals.")
    A("  Isolation Forest features: " + ", ".join(cfg.if_features) +
      f" (trailing windows: {cfg.roll_short} and {cfg.roll_long} records per machine).")
    A("  Detection is unsupervised: no labels are used for fitting or thresholds.")
    A("  Residual-z rows with z-flag AND Isolation-Forest flag AND residual>0 are anomalies"
      if cfg.combine_mode == "intersection" and cfg.direction == "increase" else
      f"  Combination rule: {cfg.combine_mode}; direction: {cfg.direction}.")
    A("  Severity (on |residual z|): low >= %.1f, medium >= %.1f, high >= %.1f." %
      (cfg.z_threshold, cfg.z_medium, cfg.z_high))

    A("\n3. CONFIGURATION")
    for k in ("baseline_strategy", "n_splits", "z_threshold", "z_medium", "z_high", "contamination",
              "combine_mode", "direction", "window_freq", "window_z_threshold", "event_gap_intervals",
              "emission_factor_kg_per_kwh", "random_state"):
        A(f"  {k}: {getattr(cfg, k)}")

    A("\n4. BASELINE QUALITY (scored records only)")
    m = out.baseline.metrics if out.baseline else {}
    if m:
        A(f"  scored records: {m.get('n_scored')}   R2 = {_f(m.get('r2'))}   MAE = {_f(m.get('mae'),3)} kWh   "
          f"RMSE = {_f(m.get('rmse'),3)} kWh   MAE = {_f(m.get('mae_pct_of_mean_actual'),1)}% of mean actual")
        if "r2_normal_rows" in m:
            A(f"  on rows labelled fault_flag=0: R2 = {_f(m['r2_normal_rows'])}   MAE = {_f(m['mae_normal_rows'],3)} kWh")
        A("  Project gate (R2 >= 0.90, MAE < 5%): "
          + ("MET" if m.get("r2", 0) >= 0.90 and m.get("mae_pct_of_mean_actual", 99) < 5 else
             "NOT MET on this dataset - baseline noise is high, which limits detection sensitivity."))
    else:
        A("  not available (external expected values or median fallback without metrics)")

    A("\n5. DETECTION RESULTS")
    sc = r[r["detection_status"] != "not_scored"]
    A(f"  scored records: {len(sc)}   anomalies: {int(r['is_anomaly'].sum())} "
      f"({100*r['is_anomaly'].mean():.2f}% of all records)   not scored: {int((r['detection_status']=='not_scored').sum())}")
    A("  severity counts: " + str(r.loc[r['is_anomaly'], 'severity'].value_counts().to_dict()))
    A(f"  anomalous events (grouped): {r.loc[r['event_id'] != '', 'event_id'].nunique()}")
    A(f"  total excess energy in flagged records: {_f(r.loc[r['is_anomaly'],'residual_kWh'].sum(),1)} kWh  "
      f"= {_f(r.loc[r['is_anomaly'],'excess_co2e_kg'].sum(),1)} kg CO2e")
    if len(out.drilldown):
        A("  drill-down (largest excess CO2e at each level): " +
          " > ".join(f"{x.level}:{x.node}" for x in out.drilldown.itertuples()))
    A(f"  sustained-deviation windows flagged ({cfg.window_freq}): {len(out.window_anomalies)}")

    A("\n6. VALIDATION AGAINST THE INJECTED-FAULT LABELS (evaluation only)")
    e = out.evaluation
    if not e.get("available"):
        A(f"  not available: {e.get('reason')}")
    else:
        A(f"  record level : recall = {_f(e['recall'])}  precision = {_f(e['precision'])}  "
          f"false-positive rate = {_f(e['false_positive_rate'])}")
        A(f"                 TP={e['tp']} FN={e['fn']} FP={e['fp']} TN={e['tn']}")
        A(f"  ranking power: ROC-AUC residual z = {_f(e.get('roc_auc_residual_z'),3)}; "
          f"ROC-AUC Isolation-Forest score = {_f(e.get('roc_auc_if_score'),3)}")
        A(f"  fault episodes detected (>=1 flagged record): {int(e['events']['detected'].sum())} of {len(e['events'])}")
        for x in e["events"].itertuples():
            A(f"    - {x.machine} {x.fault_type:<20s} {x.start} -> {x.end}  records={x.n_records:>3}  "
              f"flagged={x.n_flagged:>3}  detected={x.detected}")
        if "localization_hits_at_k" in e:
            A(f"  machine localization: top-{e['localization_k']} machines by excess CO2e = {e['top_machines_by_excess']}; "
              f"truly faulty = {e['faulty_machines']}; hits = {e['localization_hits_at_k']}/{e['localization_k']}")
        if "fault_windows_total" in e:
            A(f"  fault windows ({cfg.window_freq}) flagged: {e['fault_windows_flagged']}/{e['fault_windows_total']}; "
              f"window precision = {_f(e.get('window_precision'),2)} ({len(out.window_anomalies)} windows flagged in total)")
        A("  Project gates (recall >= 95%%, FPR < 3%%): recall %s, FPR %s" %
          ("MET" if e["gate_recall_ge_95"] else "NOT MET", "MET" if e["gate_fpr_lt_3pct"] else "NOT MET"))

    A("\n7. LIMITATIONS & HONEST CAVEATS")
    A("  * A flag is a STATISTICAL anomaly (energy use unlike the baseline). It is not a confirmed")
    A("    environmental incident; confirmation needs engineering review.")
    A("  * Localization stops at the available hierarchy (plant/building/department/machine) and time.")
    A("    Within a machine, no sub-component information exists, so none is claimed.")
    A("  * The data contain only 4 injected fault episodes (164 records); metrics are indicative only.")
    A("  * Injected faults raise consumption by a small amount relative to row-level noise of the")
    A("    baseline, so per-record recall is far below the 95% project target; detection is much")
    A("    stronger at episode / window level (see sections 5 and 6).")
    A("  * Blocked out-of-fold baseline trains on blocks that can lie in the future of the scored block;")
    A("    use baseline_strategy='chronological_holdout' for a strict no-lookahead check (scores only")
    A("    the final 20% of the timeline).")
    A("  * Isolation Forest and robust z-scale are fitted on the same period they score (unsupervised).")
    A("  * previous_hour_kwh was verified as past-only and kept; its name suggests an hour but it is the")
    A("    sum of the previous 4 fifteen-minute intervals.")

    A("\n8. EXECUTION SUMMARY")
    A("  timings (s): " + ", ".join(f"{k}={v:.2f}" for k, v in out.timings.items()))
    A("  warnings: " + (" | ".join(out.warnings) if out.warnings else "none"))
    A("  files: " + ", ".join(Path(p).name for p in paths.values()))
    return "\n".join(L) + "\n"

"""
evaluation.py - check detections against the injected-fault labels, IF they exist.

The labels (fault_flag / fault_type) are used ONLY here, never for training, thresholds
fitting inside the detector, or flagging. With only 4 injected fault episodes in the
supplied data, metrics are indicative, not statistically strong.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import Config


def _fault_events(res: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    d = res.sort_values([cfg.machine_col, cfg.timestamp_col]).copy()
    d["_pos"] = d.groupby(cfg.machine_col).cumcount()
    f = d[d["_label"] == 1]
    rows = []
    for mach, g in f.groupby(cfg.machine_col):
        n = (g["_pos"].diff().fillna(np.inf) > 1).cumsum()
        for _, e in g.groupby(n):
            rows.append({"machine": mach, "start": e[cfg.timestamp_col].min(),
                         "end": e[cfg.timestamp_col].max(), "n_records": len(e),
                         "fault_type": e[cfg.label_type_col].dropna().iloc[0]
                         if cfg.label_type_col in e and e[cfg.label_type_col].notna().any() else "",
                         "n_flagged": int(e["is_anomaly"].sum()),
                         "detected": bool(e["is_anomaly"].any())})
    return pd.DataFrame(rows)


def evaluate_against_labels(res: pd.DataFrame, cfg: Config, window_anoms: pd.DataFrame,
                            summary: pd.DataFrame) -> Dict:
    if cfg.label_flag_col not in res.columns:
        return {"available": False, "reason": f"column '{cfg.label_flag_col}' not in data"}
    r = res.copy()
    r["_label"] = pd.to_numeric(r[cfg.label_flag_col], errors="coerce").fillna(0).astype(int)
    s = r[r["detection_status"] != "not_scored"]
    if s["_label"].sum() == 0:
        return {"available": False, "reason": "no positive labels among scored rows"}
    tp = int(((s["_label"] == 1) & s["is_anomaly"]).sum())
    fn = int(((s["_label"] == 1) & ~s["is_anomaly"]).sum())
    fp = int(((s["_label"] == 0) & s["is_anomaly"]).sum())
    tn = int(((s["_label"] == 0) & ~s["is_anomaly"]).sum())
    m: Dict = {"available": True, "n_scored": len(s), "tp": tp, "fn": fn, "fp": fp, "tn": tn,
               "recall": tp / (tp + fn) if tp + fn else np.nan,
               "precision": tp / (tp + fp) if tp + fp else np.nan,
               "false_positive_rate": fp / (fp + tn) if fp + tn else np.nan}
    try:
        m["roc_auc_residual_z"] = float(roc_auc_score(s["_label"], s["residual_z"].fillna(0)))
        if s["anomaly_score"].notna().any():
            m["roc_auc_if_score"] = float(roc_auc_score(s["_label"], s["anomaly_score"].fillna(s["anomaly_score"].min())))
    except ValueError:
        pass
    ev = _fault_events(s, cfg)
    m["events"] = ev
    m["event_recall"] = float(ev["detected"].mean()) if len(ev) else np.nan

    # machine-level localization: do the top-ranked machines contain the faulty ones?
    faulty = set(s.loc[s["_label"] == 1, cfg.machine_col].unique())
    m["faulty_machines"] = sorted(faulty)
    if len(summary):
        mach = summary[(summary["level"] == cfg.machine_col.replace("_id", "")) & (summary["time_window"] == "ALL")]
        mach = mach[mach["n_anomalies"] > 0].sort_values("total_excess_co2e_kg", ascending=False)
        topk = list(mach[cfg.machine_col].head(len(faulty)))
        m["top_machines_by_excess"] = topk
        m["localization_hits_at_k"] = len(set(topk) & faulty)
        m["localization_k"] = len(faulty)
    # window-level
    if len(window_anoms):
        m["window_anomalies_flagged"] = int(len(window_anoms))
        wk = set(zip(window_anoms[cfg.machine_col], pd.to_datetime(window_anoms["window_start"])))
        s2 = s.copy()
        s2["_w"] = s2[cfg.timestamp_col].dt.floor(cfg.window_freq)
        fw = s2[s2["_label"] == 1].groupby([cfg.machine_col, "_w"]).size().index
        hit = sum((a, b) in wk for a, b in fw)
        m["fault_windows_total"] = len(fw)
        m["fault_windows_flagged"] = int(hit)
        m["window_precision"] = hit / len(wk) if len(wk) else np.nan
    gate_recall, gate_fpr = 0.95, 0.03
    m["gate_recall_ge_95"] = bool(m["recall"] >= gate_recall)
    m["gate_fpr_lt_3pct"] = bool(m["false_positive_rate"] < gate_fpr)
    return m

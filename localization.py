"""
localization.py - turn row-level flags into events, incidents and ranked locations.

What can be localized depends on the columns present. For the supplied dataset the
facility tree is Plant -> Building -> Department -> Machine, with 15-minute timestamps,
so incidents can be traced to a machine and a time window. Localization here means
"where the statistical deviation is concentrated"; it does not prove physical cause
(the explainability step) and is never a confirmed environmental incident.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from anomaly_detection import SEVERITY_ORDER, MAD_TO_SIGMA
from config import Config

_SEV_INV = {v: k for k, v in SEVERITY_ORDER.items()}


def hierarchy_levels(df: pd.DataFrame, cfg: Config) -> List[str]:
    return [c for c in cfg.hierarchy if c in df.columns and df[c].notna().any()]


# ------------------------------------------------------------------------- events
def assign_events(res: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Group consecutive anomalous records of one machine into an event_id.
    Records <= cfg.event_gap_intervals positions apart (in that machine's own series)
    belong to the same event."""
    res = res.copy()
    res["event_id"] = ""
    anom = res[res["is_anomaly"]]
    if anom.empty:
        return res
    ordered = res.sort_values([cfg.machine_col, cfg.timestamp_col])
    ordered["_pos"] = ordered.groupby(cfg.machine_col).cumcount()
    a = ordered.loc[ordered["is_anomaly"]]
    for mach, grp in a.groupby(cfg.machine_col):
        gap = grp["_pos"].diff().fillna(np.inf) > cfg.event_gap_intervals
        n = gap.cumsum()
        first_ts = grp[cfg.timestamp_col].groupby(n).transform("min")
        ids = [f"EVT-{mach}-{t:%Y%m%d%H%M}" for t in first_ts]
        res.loc[grp.index, "event_id"] = ids
    return res


def build_incidents(res: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """detected_incidents table: anomalies only, ALL columns of the results retained
    (identifiers, feature values, residuals, scores) plus incident_id / event_id first."""
    inc = res[res["is_anomaly"]].copy()
    if inc.empty:
        inc.insert(0, "incident_id", pd.Series(dtype=str))
        return inc
    inc.insert(0, "incident_id", [
        f"INC-{m}-{t:%Y%m%d%H%M}" for m, t in zip(inc[cfg.machine_col], inc[cfg.timestamp_col])])
    return inc.sort_values([cfg.timestamp_col, cfg.machine_col]).reset_index(drop=True)


# ------------------------------------------------------------------- aggregation
def _agg_group(g: pd.DataFrame, cfg: Config) -> pd.Series:
    scored = g[g["detection_status"] != "not_scored"]
    a = scored[scored["is_anomaly"]]
    sev = a["severity"].map(SEVERITY_ORDER).max() if len(a) else 0
    return pd.Series({
        "n_records_scored": len(scored),
        "n_anomalies": len(a),
        "anomaly_rate": len(a) / len(scored) if len(scored) else np.nan,
        "total_excess_kwh": a["residual_kWh"].sum() if len(a) else 0.0,
        "total_excess_co2e_kg": a["excess_co2e_kg"].sum() if len(a) else 0.0,
        "net_residual_kwh_all_scored": scored["residual_kWh"].sum() if len(scored) else np.nan,
        "mean_pct_deviation_anomalies": a["pct_deviation"].mean() if len(a) else np.nan,
        "mean_residual_z_anomalies": a["residual_z"].mean() if len(a) else np.nan,
        "max_residual_z": a["residual_z"].max() if len(a) else np.nan,
        "max_severity": _SEV_INV[int(sev)] if len(a) else "none",
        "first_anomaly_ts": a[cfg.timestamp_col].min() if len(a) else pd.NaT,
        "last_anomaly_ts": a[cfg.timestamp_col].max() if len(a) else pd.NaT,
    })


def summarize_locations(res: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Long table: one row per (level, location, time_window). time_window is 'ALL' for the
    whole period or the day start (cfg.daily_freq). Daily rows are kept only if they contain
    anomalies. `rank_in_level` ranks locations by total excess CO2e within level+time_window."""
    levels = hierarchy_levels(res, cfg)
    keys_all = list(cfg.hierarchy)
    rows = []
    res = res.copy()
    res["_day"] = res[cfg.timestamp_col].dt.floor(cfg.daily_freq).dt.strftime("%Y-%m-%d")
    for i, lvl in enumerate(levels):
        keys = levels[: i + 1]
        for tw, frame_key in (("ALL", None), ("DAILY", "_day")):
            gkeys = keys + ([frame_key] if frame_key else [])
            recs = []
            for gk, g in res.groupby(gkeys, dropna=False):
                gk = gk if isinstance(gk, tuple) else (gk,)
                s = _agg_group(g, cfg)
                if frame_key and s["n_anomalies"] == 0:
                    continue
                rec = dict(zip(gkeys, gk))
                rec.update(s.to_dict())
                recs.append(rec)
            if not recs:
                continue
            t = pd.DataFrame(recs)
            t["level"] = lvl.replace("_id", "")
            t["time_window"] = t.pop("_day") if frame_key else "ALL"
            rows.append(t)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    for c in keys_all:
        if c not in out.columns:
            out[c] = np.nan
    out["rank_in_level"] = np.nan
    has = out["n_anomalies"] > 0
    out.loc[has, "rank_in_level"] = out[has].groupby(["level", "time_window"])["total_excess_co2e_kg"] \
        .rank(ascending=False, method="min")
    lead = ["level"] + keys_all + ["time_window", "rank_in_level"]
    rest = [c for c in out.columns if c not in lead]
    out = out[lead + rest]
    order = {l.replace("_id", ""): i for i, l in enumerate(keys_all)}
    out["_o"] = out["level"].map(order)
    out["_w"] = (out["time_window"] != "ALL").astype(int)
    out = out.sort_values(["_w", "_o", "time_window", "rank_in_level"], na_position="last") \
        .drop(columns=["_o", "_w"]).reset_index(drop=True)
    return out


# ------------------------------------------------------------ sustained windows
def detect_window_anomalies(res: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Machine x time-window check for SUSTAINED excess. For each machine the mean residual
    z-score of every window (cfg.window_freq) is compared with that machine's other windows
    (robust median/MAD). Windows with too few records are ignored (a window holding one
    record would give a meaningless mean)."""
    d = res[(res["detection_status"] != "not_scored")].copy()
    if d.empty:
        return pd.DataFrame()
    d["window_start"] = d[cfg.timestamp_col].dt.floor(cfg.window_freq)
    win_len = pd.Timedelta(cfg.window_freq)
    out = []
    for mach, g in d.groupby(cfg.machine_col):
        step = g[cfg.timestamp_col].sort_values().diff().median()
        if pd.isna(step) or step <= pd.Timedelta(0):
            continue
        need = max(1, int(cfg.window_min_fraction * (win_len / step)))
        w = g.groupby("window_start").agg(
            n_records_scored=("residual_z", "size"),
            window_mean_residual_z=("residual_z", "mean"),
            window_excess_kwh=("residual_kWh", "sum"),
            window_excess_co2e_kg=("excess_co2e_kg", "sum"),
            n_row_anomalies=("is_anomaly", "sum"))
        w = w[w["n_records_scored"] >= need]
        if len(w) < cfg.window_min_windows_per_machine:
            continue
        med = w["window_mean_residual_z"].median()
        sig = (w["window_mean_residual_z"] - med).abs().median() * MAD_TO_SIGMA
        if sig <= 1e-9:
            continue
        w["window_z"] = (w["window_mean_residual_z"] - med) / sig
        w = w[(w["window_z"] > cfg.window_z_threshold) & (w["window_excess_kwh"] > 0)].reset_index()
        for c in reversed(cfg.hierarchy):
            if c in g.columns:
                w.insert(0, c, g[c].iloc[0])
        w.insert(0, "level", "machine_window")
        w["time_window"] = w["window_start"].dt.strftime("%Y-%m-%d %H:%M")
        w["window_length"] = cfg.window_freq
        out.append(w)
    if not out:
        return pd.DataFrame()
    r = pd.concat(out, ignore_index=True)
    return r.sort_values("window_excess_co2e_kg", ascending=False).reset_index(drop=True)


# -------------------------------------------------------------------- drill-down
def drill_down(res: pd.DataFrame, cfg: Config, top_k: int = 1) -> pd.DataFrame:
    """Tree traversal Plant -> Building -> Department -> Machine -> day: at each level follow
    the child(ren) with the largest total anomalous excess CO2e. Returns the traced path(s)."""
    levels = hierarchy_levels(res, cfg)
    a = res[res["is_anomaly"]].copy()
    if a.empty:
        return pd.DataFrame(columns=["level", "node", "parent", "n_anomalies",
                                     "total_excess_co2e_kg", "share_of_parent"])
    a["_day"] = a[cfg.timestamp_col].dt.floor(cfg.daily_freq).dt.strftime("%Y-%m-%d")
    rows = []
    frontier = [("", a, 1.0)]       # (path label, subset, rank-order)
    for lvl in levels + ["_day"]:
        nxt = []
        for label, sub, _ in frontier:
            tot = sub["excess_co2e_kg"].sum()
            top = sub.groupby(lvl)["excess_co2e_kg"].sum().sort_values(ascending=False).head(top_k)
            for node, val in top.items():
                s2 = sub[sub[lvl] == node]
                rows.append({"level": "day" if lvl == "_day" else lvl.replace("_id", ""),
                             "node": node, "parent": label or "FACILITY",
                             "n_anomalies": len(s2), "total_excess_co2e_kg": float(val),
                             "share_of_parent": float(val / tot) if tot else np.nan})
                nxt.append((f"{label}/{node}" if label else str(node), s2, 1.0))
        frontier = nxt
    return pd.DataFrame(rows)

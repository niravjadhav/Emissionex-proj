"""
anomaly_detection.py - deviation maths, residual z-scores and Isolation Forest.

A flagged record is a STATISTICAL anomaly: its energy use deviates from what the baseline
expected for that machine/time/state. It is NOT a confirmed environmental incident.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from config import Config

MAD_TO_SIGMA = 1.4826   # makes MAD comparable to a standard deviation for normal data
SEVERITY_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3}


# ------------------------------------------------------------------- deviation maths
def compute_deviation(df: pd.DataFrame, expected: pd.Series, cfg: Config) -> pd.DataFrame:
    """Add expected_kWh, residual_kWh (= actual - expected), abs / % deviation and CO2e."""
    out = df.copy()
    actual = out[cfg.target_col]
    out[cfg.expected_col] = expected.reindex(out.index).astype(float)
    out["residual_kWh"] = actual - out[cfg.expected_col]
    out["abs_deviation_kWh"] = out["residual_kWh"].abs()
    # % deviation is only defined when the expected value is large enough to be meaningful
    ok = out[cfg.expected_col] >= cfg.pct_min_expected_kwh
    out["pct_deviation"] = np.where(ok, 100.0 * out["residual_kWh"] / out[cfg.expected_col].where(ok), np.nan)
    f = cfg.emission_factor_kg_per_kwh
    if cfg.actual_co2e_col in out.columns:
        out["actual_co2e_kg"] = out[cfg.actual_co2e_col]
    else:
        out["actual_co2e_kg"] = actual * f
    out["expected_co2e_kg"] = out[cfg.expected_col] * f
    out["excess_co2e_kg"] = out["residual_kWh"] * f      # same factor on both sides
    return out


def robust_stats(residual: pd.Series, groups: Optional[pd.Series], min_group: int) -> Dict:
    """Median / MAD-sigma of the residual per group (+ global fallback)."""
    def one(s: pd.Series):
        s = s.dropna()
        if len(s) == 0:
            return None
        med = float(s.median())
        sig = float((s - med).abs().median() * MAD_TO_SIGMA)
        if sig <= 1e-9:
            sig = float(s.std(ddof=0)) if len(s) > 1 else 0.0
        return med, sig
    stats = {"global": one(residual) or (0.0, 0.0), "groups": {}}
    if groups is not None:
        for k, s in residual.groupby(groups):
            if s.notna().sum() >= min_group:
                r = one(s)
                if r:
                    stats["groups"][k] = r
    return stats


def apply_zscore(residual: pd.Series, groups: Optional[pd.Series], stats: Dict) -> pd.Series:
    med = pd.Series(stats["global"][0], index=residual.index, dtype=float)
    sig = pd.Series(stats["global"][1], index=residual.index, dtype=float)
    if groups is not None:
        for k, (m, s) in stats["groups"].items():
            msk = groups == k
            med[msk], sig[msk] = m, s
    z = (residual - med) / sig.where(sig > 1e-9)
    return z                                   # NaN where sigma is 0 or residual missing


# ----------------------------------------------------------------------- detector
class AnomalyDetector:
    """fit() learns per-machine residual scale + an Isolation Forest;
    score() returns flags/scores. Fit on a reference period and score new data, or fit and
    score the same table (unsupervised - no labels are used)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.stats_: Optional[Dict] = None
        self.iso_: Optional[IsolationForest] = None
        self.if_skipped_reason_: Optional[str] = None
        self.warnings_: List[str] = []

    # -- feature construction ----------------------------------------------------
    def _add_z_features(self, dev: pd.DataFrame) -> pd.DataFrame:
        cfg = self.cfg
        d = dev.copy()
        groups = d[cfg.machine_col] if cfg.machine_col in d.columns else None
        d["residual_z"] = apply_zscore(d["residual_kWh"], groups, self.stats_)
        # trailing (causal) means over each machine's own recent records
        order = d.sort_values([cfg.machine_col, cfg.timestamp_col]).index
        zs = d.loc[order, "residual_z"]
        gk = d.loc[order, cfg.machine_col]
        for name, w in (("residual_z_roll4", cfg.roll_short), ("residual_z_roll12", cfg.roll_long)):
            d.loc[order, name] = zs.groupby(gk).transform(lambda s, w=w: s.rolling(w, min_periods=1).mean())
        return d

    # -- API -------------------------------------------------------------------------
    def fit(self, dev: pd.DataFrame) -> "AnomalyDetector":
        cfg = self.cfg
        groups = dev[cfg.machine_col] if cfg.machine_col in dev.columns else None
        self.stats_ = robust_stats(dev["residual_kWh"], groups, cfg.min_group_size_for_z)
        feats = self._add_z_features(dev)
        X = self._if_matrix(feats)
        valid = X.notna().all(axis=1)
        if valid.sum() < cfg.min_rows_for_if:
            self.if_skipped_reason_ = (f"only {int(valid.sum())} scoreable rows (< {cfg.min_rows_for_if}); "
                                       "Isolation Forest skipped, z-score rule only")
            self.warnings_.append(self.if_skipped_reason_)
            self.iso_ = None
            return self
        self.iso_ = IsolationForest(n_estimators=cfg.n_estimators_if, contamination=cfg.contamination,
                                    random_state=cfg.random_state, n_jobs=1).fit(X[valid])
        return self

    def _if_matrix(self, feats: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in self.cfg.if_features if c in feats.columns]
        return feats[cols]

    def score(self, dev: pd.DataFrame) -> pd.DataFrame:
        """Return `dev` + residual_z*, anomaly_score, if_flag, z_flag, direction, is_anomaly,
        severity, detection_status."""
        if self.stats_ is None:
            raise RuntimeError("Call fit() before score().")
        cfg = self.cfg
        d = self._add_z_features(dev)
        X = self._if_matrix(d)
        valid = d["residual_kWh"].notna() & d["residual_z"].notna() & X.notna().all(axis=1)

        d["anomaly_score"] = np.nan
        d["if_flag"] = False
        if self.iso_ is not None and valid.any():
            d.loc[valid, "anomaly_score"] = -self.iso_.score_samples(X[valid])     # higher = more anomalous
            d.loc[valid, "if_flag"] = self.iso_.predict(X[valid]) == -1

        if cfg.direction == "increase":
            z_hit = d["residual_z"] > cfg.z_threshold
        else:
            z_hit = d["residual_z"].abs() > cfg.z_threshold
        d["z_flag"] = (z_hit & valid).fillna(False)

        mode = cfg.combine_mode
        if self.iso_ is None:                       # IF unavailable -> z rule only
            comb = d["z_flag"]
        elif mode == "union":
            comb = d["z_flag"] | d["if_flag"]
        elif mode == "if_only":
            comb = d["if_flag"]
        elif mode == "z_only":
            comb = d["z_flag"]
        else:
            comb = d["z_flag"] & d["if_flag"]
        if cfg.direction == "increase":
            comb = comb & (d["residual_kWh"] > 0)
        d["is_anomaly"] = (comb & valid).astype(bool)

        d["direction"] = np.where(d["residual_kWh"] > 0, "increase",
                                  np.where(d["residual_kWh"] < 0, "decrease", "none"))
        sev = np.full(len(d), "none", dtype=object)
        az = d["residual_z"].abs().fillna(0).to_numpy()
        anom = d["is_anomaly"].to_numpy()
        sev[anom] = "low"
        sev[anom & (az >= cfg.z_medium)] = "medium"
        sev[anom & (az >= cfg.z_high)] = "high"
        d["severity"] = sev
        d["detection_status"] = np.where(~valid, "not_scored", np.where(d["is_anomaly"], "anomaly", "normal"))
        return d

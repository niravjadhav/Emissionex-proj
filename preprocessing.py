"""
preprocessing.py - loading, validation, cleaning and leakage checks.

Design rules
  * Every record keeps `source_row_id` = zero-based row position in the ORIGINAL csv,
    so any downstream row can be mapped back to the source file.
  * Nothing is invented: unparsable timestamps / negative energy are reported and
    excluded from scoring, never silently "repaired".
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import Config, SOURCE_ROW_ID


class SchemaError(ValueError):
    """Raised when the dataset lacks a column the module cannot work without."""


# ----------------------------------------------------------------------------- load
def load_dataset(path) -> pd.DataFrame:
    """Read the csv (low_memory=False avoids the mixed-type warning of the notebook)
    and attach `source_row_id`."""
    df = pd.read_csv(path, low_memory=False)
    return attach_source_row_id(df)


def attach_source_row_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if SOURCE_ROW_ID not in df.columns:
        df.insert(0, SOURCE_ROW_ID, np.arange(len(df), dtype=np.int64))
    return df


# ------------------------------------------------------------------------ validation
def validate_schema(df: pd.DataFrame, cfg: Config) -> Dict:
    """Check required / optional columns. Raises SchemaError if unusable."""
    required = [cfg.timestamp_col, cfg.target_col, cfg.machine_col]
    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        raise SchemaError(
            f"Required column(s) missing: {missing_required}. Found: {list(df.columns)}")
    if len(df) == 0:
        raise SchemaError("Dataset is empty.")

    optional = (cfg.categorical_features + cfg.numeric_features + cfg.hierarchy +
                [cfg.label_flag_col, cfg.label_type_col, cfg.actual_co2e_col])
    optional = list(dict.fromkeys(optional))
    missing_optional = [c for c in optional if c not in df.columns]
    hierarchy = [c for c in cfg.hierarchy if c in df.columns and df[c].notna().any()]
    return {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "missing_optional_columns": missing_optional,
        "hierarchy_levels_available": hierarchy,
        "null_counts": {c: int(v) for c, v in df.isna().sum().items() if v > 0},
        "has_labels": cfg.label_flag_col in df.columns,
    }


# --------------------------------------------------------------------------- cleaning
def clean_dataset(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, Dict]:
    """Parse types, drop unusable / duplicate rows, sort. Returns (clean_df, report)."""
    report: Dict = {"rows_in": int(len(df))}
    df = attach_source_row_id(df)

    # timestamp
    ts = pd.to_datetime(df[cfg.timestamp_col], errors="coerce")
    bad_ts = int(ts.isna().sum())
    df[cfg.timestamp_col] = ts
    if bad_ts:
        df = df.loc[ts.notna()].copy()
    report["dropped_invalid_timestamp"] = bad_ts

    # numeric coercion of target + numeric features
    coerced: Dict[str, int] = {}
    for c in dict.fromkeys([cfg.target_col] + [c for c in cfg.numeric_features if c in df.columns]
                           + ([cfg.actual_co2e_col] if cfg.actual_co2e_col in df.columns else [])):
        before = df[c].isna().sum()
        df[c] = pd.to_numeric(df[c], errors="coerce")
        n = int(df[c].isna().sum() - before)
        if n:
            coerced[c] = n
    report["non_numeric_values_set_to_nan"] = coerced

    # physically impossible energy -> NaN (kept in the table, not scored)
    neg = int((df[cfg.target_col] < 0).sum())
    if neg:
        df.loc[df[cfg.target_col] < 0, cfg.target_col] = np.nan
    report["negative_target_set_to_nan"] = neg
    report["missing_target_rows_kept_but_not_scored"] = int(df[cfg.target_col].isna().sum())

    # duplicates
    content_cols = [c for c in df.columns if c != SOURCE_ROW_ID]
    full_dup = df.duplicated(subset=content_cols, keep="first")
    report["dropped_exact_duplicates"] = int(full_dup.sum())
    df = df.loc[~full_dup]
    key_dup = df.duplicated(subset=[cfg.timestamp_col, cfg.machine_col], keep="first")
    report["dropped_duplicate_timestamp_machine"] = int(key_dup.sum())
    df = df.loc[~key_dup]

    df = df.sort_values([cfg.timestamp_col, cfg.machine_col], kind="mergesort").reset_index(drop=True)
    report["rows_out"] = int(len(df))
    return df, report


# --------------------------------------------------------------------- leakage check
def check_rolling_leakage(df: pd.DataFrame, cfg: Config, tol: float = 0.02) -> Dict[str, Dict]:
    """
    Test whether suspected lag/rolling columns were built from the TARGET including the
    current interval (target leakage) or only from earlier intervals.

    For each column c with window w, compare c against the per-machine rolling SUM of the
    target over w intervals computed (a) including the current row and (b) shifted by one.
    """
    out: Dict[str, Dict] = {}
    d = df.sort_values([cfg.machine_col, cfg.timestamp_col])
    g = d.groupby(cfg.machine_col)[cfg.target_col]
    for col, w in cfg.leak_suspect_columns.items():
        if col not in d.columns:
            continue
        incl = g.transform(lambda s: s.rolling(w, min_periods=1).sum())
        past = g.transform(lambda s: s.rolling(w, min_periods=1).sum().shift(1))
        valid = d[col].notna() & d[cfg.target_col].notna()
        f_incl = float(((d[col] - incl).abs() <= tol)[valid].mean()) if valid.any() else 0.0
        f_past = float(((d[col] - past).abs() <= tol)[valid].mean()) if valid.any() else 0.0
        if f_incl >= 0.9 and f_incl > f_past:
            verdict = "includes_current_interval"
        elif f_past >= 0.9:
            verdict = "past_only"
        else:
            verdict = "unverified"
        out[col] = {"window_intervals": w, "match_including_current": round(f_incl, 4),
                    "match_past_only": round(f_past, 4), "verdict": verdict}
    return out


def select_baseline_features(df: pd.DataFrame, cfg: Config,
                             leak_report: Dict[str, Dict]) -> Tuple[List[str], List[str], Dict[str, str]]:
    """Return (categorical, numeric, excluded{col: reason}) actually usable for the baseline."""
    excluded: Dict[str, str] = {}
    cats = []
    for c in cfg.categorical_features:
        if c in df.columns:
            cats.append(c)
    nums = []
    for c in cfg.numeric_features:
        if c not in df.columns:
            continue
        if c in cfg.leak_suspect_columns:
            verdict = leak_report.get(c, {}).get("verdict", "unverified")
            if verdict != "past_only":
                excluded[c] = f"leakage check verdict = {verdict}"
                continue
        if df[c].notna().sum() == 0:
            excluded[c] = "all values missing"
            continue
        nums.append(c)
    return cats, nums, excluded

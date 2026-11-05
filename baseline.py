"""
baseline.py - expected-emission (expected kWh) baseline.

Reuses the Member 2 XGBoost approach (same hyper-parameters family, same one-hot encoding,
same feature list) with three changes needed for a trustworthy anomaly detector:
  1. features that leak the current target are removed (see preprocessing.check_rolling_leakage);
  2. every record is scored by a model that did NOT see that record's time block
     (blocked out-of-fold, or a chronological hold-out) - otherwise residuals shrink on
     the very records we want to flag;
  3. a robust median fallback exists for datasets that are too small to train on.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from config import Config

UNKNOWN = "UNKNOWN"


# ----------------------------------------------------------------------- encoding
def encode_features(df: pd.DataFrame, categorical: List[str], numeric: List[str],
                    columns: Optional[List[str]] = None) -> pd.DataFrame:
    """One-hot encode categoricals (missing -> 'UNKNOWN') and keep numerics as float.
    If `columns` is given the result is aligned to exactly those columns (unseen
    categories are dropped, absent ones filled with 0) - this is what SHAP needs."""
    X = pd.DataFrame(index=df.index)
    for c in numeric:
        X[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    for c in categorical:
        X[c] = df[c].astype("object").where(df[c].notna(), UNKNOWN).astype(str)
    X = pd.get_dummies(X, columns=categorical, dtype=float)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0.0)
    return X


# ------------------------------------------------------------------------- result
@dataclass
class BaselineResult:
    expected: pd.Series                      # aligned to input index; NaN = not scored
    model_id: pd.Series                      # which model produced each prediction
    method: str
    categorical: List[str] = field(default_factory=list)
    numeric: List[str] = field(default_factory=list)
    feature_columns: List[str] = field(default_factory=list)   # encoded column names
    models: Dict[str, object] = field(default_factory=dict)    # model_id -> XGBRegressor
    metrics: Dict = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encoded feature matrix in the exact column order the models were trained on."""
        return encode_features(df, self.categorical, self.numeric, self.feature_columns)

    def save(self, directory) -> None:
        """Write models (XGBoost JSON) and metadata for explainability/inference."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for mid, model in self.models.items():
            model.save_model(str(directory / f"{mid}.json"))
        meta = {"method": self.method, "categorical": self.categorical, "numeric": self.numeric,
                "feature_columns": self.feature_columns, "model_ids": list(self.models),
                "notes": self.notes}
        (directory / "baseline_meta.json").write_text(json.dumps(meta, indent=2))


def load_baseline_artifacts(directory) -> BaselineResult:
    """Reload models saved by BaselineResult.save()."""
    from xgboost import XGBRegressor
    directory = Path(directory)
    meta = json.loads((directory / "baseline_meta.json").read_text())
    models = {}
    for mid in meta["model_ids"]:
        m = XGBRegressor()
        m.load_model(str(directory / f"{mid}.json"))
        models[mid] = m
    empty = pd.Series(dtype=float)
    return BaselineResult(expected=empty, model_id=empty.astype(object), method=meta["method"],
                          categorical=meta["categorical"], numeric=meta["numeric"],
                          feature_columns=meta["feature_columns"], models=models,
                          notes=meta["notes"])


# --------------------------------------------------------------------- strategies
def _new_model(cfg: Config):
    from xgboost import XGBRegressor
    return XGBRegressor(**cfg.xgb_params)


def _time_blocks(df: pd.DataFrame, cfg: Config, n_blocks: int) -> np.ndarray:
    """Contiguous blocks over UNIQUE timestamps (all machines at one time share a block)."""
    ts = df[cfg.timestamp_col]
    rank = ts.rank(method="dense").to_numpy() - 1
    n_unique = rank.max() + 1
    return np.minimum((rank * n_blocks // n_unique).astype(int), n_blocks - 1)


def median_baseline(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """Robust fallback: median target by (machine, status, hour) -> (machine, status) ->
    machine -> global. Computed in-sample (medians are insensitive to a few spikes)."""
    y = df[cfg.target_col]
    levels = [[c for c in (cfg.machine_col, "operating_status", "hour") if c in df.columns],
              [c for c in (cfg.machine_col, "operating_status") if c in df.columns],
              [c for c in (cfg.machine_col,) if c in df.columns]]
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for keys in levels:
        if not keys:
            continue
        grp = df.groupby(keys)[cfg.target_col]
        med = grp.transform("median")
        size = grp.transform("count")
        fill = out.isna() & (size >= cfg.median_min_group_size)
        out[fill] = med[fill]
    out = out.fillna(y.median())
    out[y.isna()] = out[y.isna()]          # prediction exists even when actual is missing
    return out


def _metrics(y: pd.Series, p: pd.Series, normal_mask: Optional[pd.Series]) -> Dict:
    ok = y.notna() & p.notna()
    if ok.sum() < 2:
        return {"n_scored": int(ok.sum())}
    m = {"n_scored": int(ok.sum()),
         "r2": float(r2_score(y[ok], p[ok])),
         "mae": float(mean_absolute_error(y[ok], p[ok])),
         "rmse": float(np.sqrt(np.mean((y[ok] - p[ok]) ** 2))),
         "mae_pct_of_mean_actual": float(100 * mean_absolute_error(y[ok], p[ok]) / max(y[ok].mean(), 1e-9))}
    if normal_mask is not None:
        okn = ok & normal_mask
        if okn.sum() >= 2:
            m["r2_normal_rows"] = float(r2_score(y[okn], p[okn]))
            m["mae_normal_rows"] = float(mean_absolute_error(y[okn], p[okn]))
    return m


def fit_expected_baseline(df: pd.DataFrame, cfg: Config, categorical: List[str],
                          numeric: List[str]) -> BaselineResult:
    """Produce expected kWh for every scoreable row according to cfg.baseline_strategy."""
    y = df[cfg.target_col]
    notes: List[str] = []
    normal_mask = None
    if cfg.label_flag_col in df.columns:
        normal_mask = (pd.to_numeric(df[cfg.label_flag_col], errors="coerce").fillna(0) == 0)

    strategy = cfg.baseline_strategy
    n_unique_ts = df[cfg.timestamp_col].nunique()
    if strategy != "median" and (y.notna().sum() < cfg.min_rows_for_model or n_unique_ts < cfg.n_splits * 2):
        notes.append(f"Dataset too small for XGBoost ({int(y.notna().sum())} valid rows, "
                     f"{n_unique_ts} timestamps); using median fallback baseline.")
        strategy = "median"

    expected = pd.Series(np.nan, index=df.index, dtype=float)
    model_id = pd.Series("", index=df.index, dtype=object)
    models: Dict[str, object] = {}
    X = encode_features(df, categorical, numeric)
    cols = list(X.columns)

    train_ok = y.notna()
    if cfg.exclude_labeled_faults_from_training and normal_mask is not None:
        train_ok &= normal_mask
        notes.append("Rows with fault_flag=1 were excluded from baseline training "
                     "(config.exclude_labeled_faults_from_training=True).")

    if strategy == "blocked_oof":
        blocks = _time_blocks(df, cfg, cfg.n_splits)
        for b in range(cfg.n_splits):
            te = blocks == b
            tr = (~te) & train_ok.to_numpy()
            if tr.sum() < 10 or te.sum() == 0:
                continue
            m = _new_model(cfg).fit(X.loc[tr], y[tr])
            expected.loc[te] = m.predict(X.loc[te])
            mid = f"fold_{b}"
            model_id.loc[te] = mid
            models[mid] = m
        method = (f"XGBoost blocked out-of-fold ({cfg.n_splits} contiguous time blocks; each record "
                  "is predicted by a model trained only on the OTHER blocks)")
        notes.append("Blocked OOF trains on blocks that may lie in the future of the scored block. "
                     "Use baseline_strategy='chronological_holdout' for strictly no-lookahead evaluation "
                     "(scores only the last part of the timeline).")
    elif strategy == "chronological_holdout":
        ts_rank = df[cfg.timestamp_col].rank(method="dense")
        cut = ts_rank.max() * cfg.train_fraction
        tr = ((ts_rank <= cut) & train_ok).to_numpy()
        te = (ts_rank > cut).to_numpy()
        m = _new_model(cfg).fit(X.loc[tr], y[tr])
        expected.loc[te] = m.predict(X.loc[te])
        model_id.loc[te] = "holdout"
        models["holdout"] = m
        method = (f"XGBoost chronological hold-out (train on first {cfg.train_fraction:.0%} of "
                  "timestamps, score the rest; earlier rows are NOT scored)")
    else:
        expected = median_baseline(df, cfg)
        model_id[:] = "median"
        method = "Robust median baseline by (machine, operating_status, hour) [fallback / no ML]"
        if strategy != cfg.baseline_strategy or cfg.baseline_strategy == "median":
            pass

    if cfg.fit_final_model and strategy in ("blocked_oof", "chronological_holdout") and train_ok.sum() >= 10:
        models["final"] = _new_model(cfg).fit(X.loc[train_ok], y[train_ok])
        notes.append("Model 'final' is trained on all usable rows - use it for NEW incoming data; "
                     "for the records in this run use the model named in `baseline_model_id`.")

    return BaselineResult(expected=expected, model_id=model_id, method=method,
                          categorical=categorical, numeric=numeric, feature_columns=cols,
                          models=models, metrics=_metrics(y, expected, normal_mask), notes=notes)

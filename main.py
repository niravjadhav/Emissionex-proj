"""
main.py - run the full EmissioNex workflow:  CSV -> baseline -> detection -> localization -> outputs.

    python main.py                                  # defaults from config.py
    python main.py --input data/my.csv --contamination 0.01 --z-threshold 3
    python main.py --baseline-strategy chronological_holdout
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import Config, DEFAULT_CONFIG
from pipeline import run_detection
from reporting import write_outputs


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="EmissioNex - anomaly detection & localization")
    p.add_argument("--input", type=Path, default=DEFAULT_CONFIG.input_csv)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_CONFIG.output_dir)
    p.add_argument("--contamination", type=float, default=DEFAULT_CONFIG.contamination,
                   help="Isolation Forest anomaly share (0-0.5)")
    p.add_argument("--z-threshold", type=float, default=DEFAULT_CONFIG.z_threshold)
    p.add_argument("--combine-mode", choices=["intersection", "union", "if_only", "z_only"],
                   default=DEFAULT_CONFIG.combine_mode)
    p.add_argument("--direction", choices=["increase", "both"], default=DEFAULT_CONFIG.direction)
    p.add_argument("--baseline-strategy", choices=["blocked_oof", "chronological_holdout", "median"],
                   default=DEFAULT_CONFIG.baseline_strategy)
    p.add_argument("--no-save-models", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    if not 0 < a.contamination <= 0.5:
        print("--contamination must be in (0, 0.5]", file=sys.stderr)
        return 2
    cfg = DEFAULT_CONFIG.with_overrides(
        input_csv=a.input, output_dir=a.output_dir, contamination=a.contamination,
        z_threshold=a.z_threshold, combine_mode=a.combine_mode, direction=a.direction,
        baseline_strategy=a.baseline_strategy)
    if not Path(cfg.input_csv).exists():
        print(f"Input file not found: {cfg.input_csv}", file=sys.stderr)
        return 2
    print(f"[1/3] Running detection on {cfg.input_csv} ...")
    out = run_detection(cfg.input_csv, cfg)
    print("[2/3] Writing outputs ...")
    paths = write_outputs(out, cfg.output_dir, save_models=not a.no_save_models)
    print("[3/3] Done.")
    r = out.results
    print(f"  records: {len(r)}  anomalies: {int(r['is_anomaly'].sum())}  "
          f"incidents file rows: {len(out.incidents)}  total time: {out.timings['total_s']:.1f}s")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    for w in out.warnings:
        print("  WARNING:", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())

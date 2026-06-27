#!/usr/bin/env python3
"""
Collate evaluation outputs (combined_per_scenario.csv) from multiple runs
into a single summary CSV suitable for thesis tables.

Usage examples:
  # Basic: two runs
  python scripts/evaluation/collate_eval_results.py \
    --run baseline=results/policy_eval_final \
    --run ambient_plus5=results/robustness/ambient_plus5 \
    --out results/collated/summary.csv

  # Add more runs (power x1.2, delay 1, cross-surrogate)
  python scripts/evaluation/collate_eval_results.py \
    --run power_x1p2=results/robustness/power_x1p2 \
    --run delay1=results/robustness/delay1_stress \
    --run rlrc_on_rcnn=results/cross_eval/rl_rc_on_rcnn \
    --run rcnn_on_rc=results/cross_eval/rl_rcnn_on_rc \
    --out results/collated/summary_extended.csv

Notes:
- Each input directory must contain combined_per_scenario.csv produced by policy_eval.py
- This script adds a 'run' column with the provided label.
"""

import argparse
from pathlib import Path
import sys
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser(description="Collate evaluation results into one CSV")
    ap.add_argument(
        "--run", action="append", default=[], metavar="LABEL=DIR",
        help="Input run as LABEL=PATH_TO_OUTPUT_DIR (expects combined_per_scenario.csv)"
    )
    ap.add_argument("--out", type=str, default="results/collated/summary.csv", help="Output CSV path")
    return ap.parse_args()


def main():
    args = parse_args()
    if not args.run:
        print("[ERROR] Provide at least one --run LABEL=DIR argument")
        sys.exit(2)

    rows = []
    for spec in args.run:
        if "=" not in spec:
            print(f"[WARN] Skipping malformed --run spec: {spec}")
            continue
        label, dir_path = spec.split("=", 1)
        d = Path(dir_path)
        csv_path = d / "combined_per_scenario.csv"
        if not csv_path.exists():
            print(f"[WARN] Missing combined_per_scenario.csv in {d}; skipping")
            continue
        try:
            df = pd.read_csv(csv_path)
            df.insert(0, "run", label)
            rows.append(df)
        except Exception as e:
            print(f"[WARN] Failed to read {csv_path}: {e}")

    if not rows:
        print("[ERROR] No valid inputs to collate")
        sys.exit(2)

    out_df = pd.concat(rows, ignore_index=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"[INFO] Wrote collated summary to {out_path}")


if __name__ == "__main__":
    main()

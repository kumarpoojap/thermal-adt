"""
Convert synthetic_thermal_dataset_v3.csv to parquet with proper datetime index.
"""
import argparse
import pandas as pd
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert synthetic CSV to parquet with a datetime index.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--input-csv",
        default="synthetic_thermal_dataset_v3.csv",
        help="Input CSV path.",
    )
    ap.add_argument(
        "--output-parquet",
        default=str(Path("data") / "synthetic" / "thermal_dataset.parquet"),
        help="Output parquet path.",
    )
    ap.add_argument(
        "--origin",
        default="2024-01-01",
        help="Origin date used when converting time_s to timestamp.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output parquet if it already exists.",
    )
    args = ap.parse_args()

    input_csv = Path(args.input_csv)
    output_parquet = Path(args.output_parquet)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    if output_parquet.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output parquet already exists: {output_parquet}. "
            f"Use --overwrite to replace it."
        )

    # Read CSV
    df = pd.read_csv(input_csv)

    if "time_s" not in df.columns:
        raise KeyError(
            "Expected a 'time_s' column in the CSV to build a timestamp index. "
            "If your file differs, adjust the script accordingly."
        )

    # Create datetime index from time_s
    df["timestamp"] = pd.to_datetime(df["time_s"], unit="s", origin=args.origin)
    df = df.drop("time_s", axis=1)
    df = df.set_index("timestamp")

    print(f"Loaded {len(df)} samples")
    print(f"Time range: {df.index[0]} to {df.index[-1]}")
    print(f"Columns: {df.columns.tolist()}")

    # Save to parquet
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet)
    print(f"\nSaved to: {output_parquet}")
    print(f"File size: {output_parquet.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()

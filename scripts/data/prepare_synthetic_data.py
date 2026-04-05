"""
Convert synthetic_thermal_dataset_v3.csv to parquet with proper datetime index.
"""
import pandas as pd
from pathlib import Path

# Read CSV
df = pd.read_csv('synthetic_thermal_dataset_v3.csv')

# Create datetime index from time_s (assuming start at epoch 0 for simplicity)
df['timestamp'] = pd.to_datetime(df['time_s'], unit='s', origin='2024-01-01')
df = df.drop('time_s', axis=1)
df = df.set_index('timestamp')

print(f"Loaded {len(df)} samples")
print(f"Time range: {df.index[0]} to {df.index[-1]}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst 5 rows:")
print(df.head())

# Save to parquet
output_dir = Path('artifacts')
output_dir.mkdir(exist_ok=True)
output_path = output_dir / 'synthetic_gpu_thermal.parquet'
df.to_parquet(output_path)
print(f"\nSaved to: {output_path}")
print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")

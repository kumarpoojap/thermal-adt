#!/usr/bin/env python3
"""
Train RF teacher on GPU thermal dataset (OS-agnostic Python version)
Works on Windows, Linux, and macOS
"""
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, shell=False)
    
    if result.returncode != 0:
        print(f"\n[ERROR] {description} failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    
    print(f"[SUCCESS] {description} completed successfully")
    return result

def main():
    # Ensure we're in the project root
    project_root = Path(__file__).parent
    
    print(f"Project root: {project_root}")
    print(f"Python: {sys.executable}")
    
    # Step 1: Train RF teacher
    rf_train_cmd = [
        sys.executable,
        "src/models/ml_surrogate_rack_temp.py",
        "--parquet", "artifacts/synthetic_gpu_thermal.parquet",
        "--spec", "configs/gpu_feature_target_spec.json",
        "--outdir", "artifacts/gpu_rf",
        "--target-mode", "normalized",
        "--lags", "1", "3", "5", "10",
        "--k_ahead", "10",
        "--n-estimators", "200",
        "--random-state", "42"
    ]
    
    run_command(rf_train_cmd, "Training RF Teacher on GPU Dataset")
    
    # Step 2: Export teacher bundle
    export_cmd = [
        sys.executable,
        "export_rf_teacher.py",
        "--rf-model", "artifacts/gpu_rf/model_random_forest.pkl",
        "--feature-cols", "artifacts/gpu_rf/feature_columns.json",
        "--target-cols", "artifacts/gpu_rf/targets_used.json",
        "--target-stats", "artifacts/gpu_rf/target_normalization_stats.json",
        "--feature-winsor-bounds", "artifacts/gpu_rf/feature_winsor_bounds.json",
        "--k-ahead", "10",
        "--cadence-s", "1",
        "--sample-parquet", "artifacts/synthetic_gpu_thermal.parquet",
        "--out", "artifacts/gpu_rf_teacher.pkl"
    ]
    
    run_command(export_cmd, "Exporting Teacher Bundle")
    
    # Step 3: Copy feature columns
    print(f"\n{'='*60}")
    print("Copying Feature Columns")
    print(f"{'='*60}")
    
    import shutil
    src = Path("artifacts/gpu_rf/feature_columns.json")
    dst = Path("artifacts/gpu_feature_columns.json")
    
    if src.exists():
        shutil.copy(src, dst)
        print(f"[SUCCESS] Copied {src} -> {dst}")
    else:
        print(f"[WARNING] {src} not found")
    
    # Final summary
    print(f"\n{'='*60}")
    print("[SUCCESS] Teacher Training Complete!")
    print(f"{'='*60}")
    print("\nGenerated files:")
    print("  - artifacts/gpu_rf/model_random_forest.pkl")
    print("  - artifacts/gpu_rf/feature_columns.json")
    print("  - artifacts/gpu_rf/metrics_summary.csv")
    print("  - artifacts/gpu_rf_teacher.pkl (teacher bundle)")
    print("  - artifacts/gpu_feature_columns.json")
    print("\nNext steps:")
    print("  1. Run PINN dev-run:")
    print(f"     {sys.executable} -m training.train_pinn_hybrid --config configs/train_gpu_pinn.yaml --dev-run")
    print("\n  2. Run full training:")
    print(f"     {sys.executable} -m training.train_pinn_hybrid --config configs/train_gpu_pinn.yaml")

if __name__ == "__main__":
    main()

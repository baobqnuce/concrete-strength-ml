"""Predict compressive strength (MPa) for new concrete mixes with the trained model.

Usage (from the repository root, after `python -m src.train`):
    python -m src.predict examples/new_mixes.csv
    python -m src.predict examples/new_mixes.csv --out predictions.csv

Input CSV columns (kg per m^3 of concrete, age in days):
    cement, slag, ash, water, superplastic, coarseagg, fineagg, age
An optional `strength` column (measured, MPa) is used to print the error.
"""
from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd

from . import data

MODEL_PATH = data.ROOT / "models" / "best_model.joblib"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="CSV file with one mix per row")
    ap.add_argument("--out", help="optional path to save predictions")
    args = ap.parse_args()

    bundle = joblib.load(MODEL_PATH)
    df = pd.read_csv(args.csv)
    missing = [c for c in bundle["features"] if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns: {missing}")
    if (df["age"] <= 0).any():
        raise SystemExit("age must be > 0 days")
    lo, hi = data.load()[bundle["features"]].min(), data.load()[bundle["features"]].max()
    outside = ((df[bundle["features"]] < lo) | (df[bundle["features"]] > hi)).any(axis=1)

    df["predicted_strength_MPa"] = bundle["model"].predict(df[bundle["features"]].values).round(2)
    df["outside_training_range"] = outside
    print(f"Model: polynomial degree {bundle['degree']} + ridge (alpha={bundle['alpha']:.3g})")
    print(df.to_string(index=False))
    if "strength" in df.columns:
        err = df["predicted_strength_MPa"] - df["strength"]
        print(f"\nMAE on these rows: {np.abs(err).mean():.2f} MPa")
    if outside.any():
        print("\nWarning: some inputs lie outside the training range; predictions are extrapolations.")
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()

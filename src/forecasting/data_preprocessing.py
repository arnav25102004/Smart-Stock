"""
Data preprocessing for the demand forecasting experiment.

Loads the Corporación Favorita 'train.csv' from data/raw/, filters for a single
store and 4 product families, adds calendar features, performs a chronological
80/20 split, and saves processed CSVs for all downstream model scripts.

Usage:
    Place train.csv in data/raw/ (download from Kaggle:
    https://www.kaggle.com/competitions/store-sales-time-series-forecasting)
    Then run: python src/forecasting/data_preprocessing.py
"""

import os
import sys
import json
import pandas as pd
import numpy as np

RAW_PATH = "data/raw/train.csv"
PROCESSED_DIR = "data/processed"
SPLIT_META_PATH = "data/processed/split_meta.json"

STORE_NBR = 1
FAMILIES = ["GROCERY I", "DAIRY", "BEVERAGES", "CLEANING"]
RANDOM_SEED = 42


def load_raw(path):
    """Load the raw Kaggle train.csv and return a DataFrame.

    Args:
        path: Path to the raw train.csv file.

    Returns:
        DataFrame with columns: id, date, store_nbr, family, sales, onpromotion.
    """
    print(f"  Loading raw data from: {path}")
    df = pd.read_csv(path, parse_dates=["date"])
    print(f"  Raw shape: {df.shape}")
    return df


def filter_and_aggregate(df):
    """Filter to store 1 and 4 families, then aggregate daily sales per family.

    Args:
        df: Raw DataFrame from Kaggle.

    Returns:
        Pivoted DataFrame with date as index and one column per product family.
    """
    df = df[df["store_nbr"] == STORE_NBR].copy()
    df = df[df["family"].isin(FAMILIES)].copy()
    print(f"  After filtering (store={STORE_NBR}, families={FAMILIES}): {df.shape}")

    # Aggregate daily totals per family
    daily = df.groupby(["date", "family"])["sales"].sum().reset_index()
    pivot = daily.pivot(index="date", columns="family", values="sales")
    pivot.columns.name = None
    pivot = pivot.reset_index()
    return pivot


def add_calendar_features(df):
    """Add day_of_week, month, and is_weekend columns to the DataFrame.

    Args:
        df: DataFrame with a 'date' column.

    Returns:
        DataFrame with three additional calendar feature columns.
    """
    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek   # 0=Monday … 6=Sunday
    df["month"] = df["date"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def fill_missing_dates(df):
    """Fill any missing dates in the time series with 0 sales.

    Args:
        df: DataFrame with a 'date' column.

    Returns:
        DataFrame with a complete daily date range (no gaps).
    """
    full_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    df = df.set_index("date").reindex(full_range, fill_value=0).reset_index()
    df = df.rename(columns={"index": "date"})
    # Re-derive calendar features after filling
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def chronological_split(df, train_ratio=0.8):
    """Split the DataFrame chronologically (no shuffling).

    Args:
        df: DataFrame sorted by date.
        train_ratio: Fraction of dates to use for training.

    Returns:
        Tuple (train_df, test_df).
    """
    split_idx = int(len(df) * train_ratio)
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test


def print_statistics(df, train, test):
    """Print dataset statistics to the console for reproducibility records."""
    print(f"\n  === Dataset Statistics ===")
    print(f"  Total rows       : {len(df)}")
    print(f"  Date range       : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Train rows       : {len(train)}  ({train['date'].min().date()} → {train['date'].max().date()})")
    print(f"  Test rows        : {len(test)}   ({test['date'].min().date()} → {test['date'].max().date()})")
    print()
    for fam in FAMILIES:
        if fam in df.columns:
            print(f"  {fam:15s} — mean: {df[fam].mean():.2f}, std: {df[fam].std():.2f}, "
                  f"min: {df[fam].min():.0f}, max: {df[fam].max():.0f}")


def main():
    print("=" * 60)
    print("STEP 1: Data Preprocessing")
    print("=" * 60)

    if not os.path.exists(RAW_PATH):
        print(f"\n[ERROR] Raw data file not found: {RAW_PATH}")
        print("  Please download the dataset from Kaggle:")
        print("  https://www.kaggle.com/competitions/store-sales-time-series-forecasting")
        print("  and place train.csv in data/raw/")
        sys.exit(1)

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("\n  [1/5] Loading raw CSV...")
    df = load_raw(RAW_PATH)

    print("\n  [2/5] Filtering and aggregating by family...")
    df = filter_and_aggregate(df)

    print("\n  [3/5] Adding calendar features...")
    df = add_calendar_features(df)

    print("\n  [4/5] Filling missing dates...")
    df = fill_missing_dates(df)
    df = df.sort_values("date").reset_index(drop=True)

    print("\n  [5/5] Splitting train/test (80/20 chronological)...")
    train, test = chronological_split(df, train_ratio=0.8)
    print_statistics(df, train, test)

    # Save processed files
    train_path = os.path.join(PROCESSED_DIR, "train_data.csv")
    test_path = os.path.join(PROCESSED_DIR, "test_data.csv")
    full_path = os.path.join(PROCESSED_DIR, "full_data.csv")

    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    df.to_csv(full_path, index=False)

    # Save split metadata so all model scripts use identical boundaries
    meta = {
        "split_index": int(len(df) * 0.8),
        "train_start": str(train["date"].min().date()),
        "train_end": str(train["date"].max().date()),
        "test_start": str(test["date"].min().date()),
        "test_end": str(test["date"].max().date()),
        "families": FAMILIES,
        "store_nbr": STORE_NBR,
        "random_seed": RANDOM_SEED,
        "feature_columns": ["day_of_week", "month", "is_weekend"],
        "target_columns": FAMILIES
    }
    with open(SPLIT_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Saved: {train_path}")
    print(f"  Saved: {test_path}")
    print(f"  Saved: {full_path}")
    print(f"  Saved: {SPLIT_META_PATH}")
    print("\n  [DONE] Data preprocessing complete.")


if __name__ == "__main__":
    main()

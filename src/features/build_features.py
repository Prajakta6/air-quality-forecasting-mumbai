import pandas as pd
import numpy as np
from pathlib import Path


# -------------------------
# Paths
# -------------------------
DATA_IN = Path("data/processed/mumbai_aqi_daily.csv")
DATA_OUT = Path("data/processed/mumbai_features.csv")


def build_features():
    # Load data
    df = pd.read_csv(DATA_IN)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # -------------------------
    # Time-based features
    # -------------------------
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # -------------------------
    # Lag features (AQI)
    # -------------------------
    df["aqi_lag_1"] = df["aqi"].shift(1)
    df["aqi_lag_7"] = df["aqi"].shift(7)
    df["aqi_lag_14"] = df["aqi"].shift(14)

    # -------------------------
    # Rolling statistics
    # -------------------------
    df["aqi_roll_7"] = df["aqi"].rolling(window=7).mean()
    df["aqi_roll_14"] = df["aqi"].rolling(window=14).mean()
    df["aqi_roll_30"] = df["aqi"].rolling(window=30).mean()

    # -------------------------
    # Pollutant-based features
    # -------------------------
    df["pm_ratio"] = df["pm25"] / (df["pm10"] + 1)

    # -------------------------
    # Target variable (next-day AQI)
    # -------------------------
    df["target_aqi"] = df["aqi"].shift(-1)

    # -------------------------
    # Drop rows with NaNs (from lags/rolling)
    # -------------------------
    df = df.dropna().reset_index(drop=True)

    # -------------------------
    # Save features
    # -------------------------
    df.to_csv(DATA_OUT, index=False)
    print(f"✅ Feature engineering complete. Saved to {DATA_OUT}")


if __name__ == "__main__":
    build_features()
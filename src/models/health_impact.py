import pandas as pd
from pathlib import Path


# -------------------------
# Paths
# -------------------------
DATA_PATH = Path("data/processed/mumbai_features.csv")
OUT_PATH = Path("data/processed/mumbai_health_impact.csv")


def map_health_risk(aqi):
    if aqi <= 50:
        return "Good – Minimal health impact"
    elif aqi <= 100:
        return "Satisfactory – Minor breathing discomfort"
    elif aqi <= 200:
        return "Moderate – Discomfort to sensitive groups"
    elif aqi <= 300:
        return "Poor – Discomfort to most people"
    elif aqi <= 400:
        return "Very Poor – Respiratory illness on prolonged exposure"
    else:
        return "Severe – Serious health effects"


def main():
    # Load data
    df = pd.read_csv(DATA_PATH)

    # Map AQI to health risk
    df["health_risk"] = df["target_aqi"].apply(map_health_risk)

    # Save output
    df.to_csv(OUT_PATH, index=False)
    print("✅ Health impact analysis completed")
    print(df[["target_aqi", "health_risk"]].head())


if __name__ == "__main__":
    main()
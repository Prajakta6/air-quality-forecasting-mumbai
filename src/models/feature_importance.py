import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path


# -------------------------
# Paths
# -------------------------
DATA_PATH = Path("data/processed/mumbai_features.csv")
MODEL_PATH = Path("models/random_forest.pkl")
OUT_PATH = Path("models/feature_importance.csv")


def main():
    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Separate features and target
    X = df.drop(columns=["date", "aqi_bucket", "target_aqi"])

    # Load trained Random Forest model
    rf = joblib.load(MODEL_PATH)

    # Extract feature importance
    importance = rf.feature_importances_

    # Create DataFrame
    feat_imp = pd.DataFrame({
        "feature": X.columns,
        "importance": importance
    }).sort_values(by="importance", ascending=False)

    # Save importance values
    feat_imp.to_csv(OUT_PATH, index=False)

    # -------------------------
    # Plot feature importance
    # -------------------------
    plt.figure(figsize=(10, 6))
    plt.barh(feat_imp["feature"], feat_imp["importance"])
    plt.gca().invert_yaxis()
    plt.title("Feature Importance - Random Forest (Mumbai AQI)")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.show()

    print("✅ Feature importance analysis completed")
    print(feat_imp.head(10))


if __name__ == "__main__":
    main()
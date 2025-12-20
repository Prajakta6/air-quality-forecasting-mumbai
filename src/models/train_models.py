import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


# -------------------------
# Paths
# -------------------------
DATA_PATH = Path("data/processed/mumbai_features.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def evaluate_model(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred)
    }


def main():
    # Load data
    df = pd.read_csv(DATA_PATH)

    # Features & target
    X = df.drop(columns=["date", "aqi_bucket", "target_aqi"])
    y = df["target_aqi"]

    # Train-test split (time-aware)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    metrics = []

    # -------------------------
    # Linear Regression
    # -------------------------
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    lr_metrics = evaluate_model(y_test, y_pred_lr)
    lr_metrics["Model"] = "Linear Regression"
    metrics.append(lr_metrics)

    joblib.dump(lr, MODEL_DIR / "linear_regression.pkl")

    # -------------------------
    # Random Forest
    # -------------------------
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    rf_metrics = evaluate_model(y_test, y_pred_rf)
    rf_metrics["Model"] = "Random Forest"
    metrics.append(rf_metrics)

    joblib.dump(rf, MODEL_DIR / "random_forest.pkl")

    # -------------------------
    # Save metrics
    # -------------------------
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(MODEL_DIR / "model_metrics.csv", index=False)

    print("✅ Model training complete")
    print(metrics_df)


if __name__ == "__main__":
    main()
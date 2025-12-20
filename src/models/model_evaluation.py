import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = Path("data/processed/mumbai_features.csv")
MODEL_PATH = Path("models/random_forest.pkl")

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["date", "aqi_bucket", "target_aqi"])
y = df["target_aqi"]

split_idx = int(len(df) * 0.8)
X_test = X.iloc[split_idx:]
y_test = y.iloc[split_idx:]

model = joblib.load(MODEL_PATH)
y_pred = model.predict(X_test)

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Actual vs Predicted AQI")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color="red")
plt.show()
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------
# Paths
# -------------------------
DATA_PATH = Path("data/processed/mumbai_aqi_daily.csv")
FEATURE_DATA_PATH = Path("data/processed/mumbai_features.csv")
MODEL_PATH = Path("models/random_forest.pkl")
FI_PATH = Path("models/feature_importance.csv")

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Mumbai AQI Forecast & Health Impact",
    layout="wide"
)

st.title("🌍 Mumbai Air Quality Forecasting Dashboard")
st.markdown(
    """
    This dashboard presents historical air quality trends, 
    next-day AQI prediction using a machine learning model, 
    and associated health risk interpretation.
    """
)

# -------------------------
# Load data
# -------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, parse_dates=["date"])

@st.cache_data
def load_features():
    return pd.read_csv(FEATURE_DATA_PATH)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

df = load_data()
features_df = load_features()
model = load_model()

# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("Navigation")
section = st.sidebar.radio(
    "Go to",
    ["Overview", "Historical AQI", "AQI Prediction", "Health Impact"]
)

# -------------------------
# Section 1: Overview
# -------------------------
if section == "Overview":
    st.subheader("📌 Project Overview")
    st.write(
        """
        - City: Mumbai  
        - Dataset: Kaggle AQI India (2015–2024)  
        - Model: Random Forest Regressor  
        - Target: Next-day AQI forecasting  
        """
    )

# -------------------------
# Section 2: Historical AQI
# -------------------------
elif section == "Historical AQI":
    st.subheader("📈 Historical AQI Trend")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["date"], df["aqi"], color="red")
    ax.set_xlabel("Year")
    ax.set_ylabel("AQI")
    ax.set_title("Mumbai AQI Over Time")
    st.pyplot(fig)

# -------------------------
# Section 3: AQI Prediction
# -------------------------
elif section == "AQI Prediction":
    st.subheader("🔮 Next-Day AQI Prediction")

    latest_row = features_df.iloc[-1:]
    X_latest = latest_row.drop(columns=["date", "aqi_bucket", "target_aqi"])

    prediction = model.predict(X_latest)[0]

    st.metric("Predicted AQI (Next Day)", f"{int(prediction)}")

# -------------------------
# Section 4: Health Impact
# -------------------------
elif section == "Health Impact":
    st.subheader("🩺 Health Impact Assessment")

    latest_row = features_df.iloc[-1:]
    X_latest = latest_row.drop(columns=["date", "aqi_bucket", "target_aqi"])
    predicted_aqi = int(model.predict(X_latest)[0])

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

    risk = map_health_risk(predicted_aqi)

    st.metric("Predicted AQI", predicted_aqi)
    st.success(f"Health Impact: {risk}")

st.markdown("---")
st.caption("MSc Data Science Project | Air Quality Forecasting – Mumbai")
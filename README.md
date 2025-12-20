# 🌍 Air Quality Forecasting & Health Impact Analysis  
### Case Study: Mumbai, India

---

## 📌 Project Overview
This project presents an end-to-end **Air Quality Forecasting and Health Impact Analysis** system using historical AQI data for **Mumbai, India**.  
The goal is to analyze air pollution trends, forecast next-day AQI using machine learning models, and translate predictions into **public health risk categories**.

The project is developed as part of an **MSc Data Science academic project** and is deployed as an interactive **Streamlit web application**.

---

## 🎯 Objectives
- Analyze historical AQI trends for Mumbai
- Identify seasonal and pollutant-driven patterns
- Engineer time-series features for AQI forecasting
- Train and evaluate machine learning models
- Map AQI predictions to health impact categories
- Deploy results using a Streamlit dashboard

---

## 📊 Dataset
- **Source:** Kaggle – Air Quality Data in India (2015–2024)
- **City:** Mumbai
- **Pollutants:** PM2.5, PM10
- **Target Variable:** Next-day AQI

---

## 🧠 Methodology
1. **Data Cleaning & Preprocessing**
   - City-level filtering (Mumbai)
   - Handling missing values
   - Daily aggregation

2. **Exploratory Data Analysis (EDA)**
   - AQI trends over time
   - Seasonal (monthly) variation
   - Pollutant correlation analysis

3. **Feature Engineering**
   - Lag features (AQI t-1, t-7, t-14)
   - Rolling averages (7, 14, 30 days)
   - Temporal features (month, weekday)
   - Pollutant ratio features

4. **Model Training & Evaluation**
   - Linear Regression (baseline)
   - Random Forest Regressor
   - Evaluation using MAE, RMSE, R²

5. **Health Impact Analysis**
   - Mapping AQI predictions to CPCB health categories

6. **Deployment**
   - Interactive dashboard built using Streamlit
   - Live web application deployment

---

## 📁 Project Structure
air_quality_project/
│
├── data/
│   ├── raw/                # Raw datasets
│   └── processed/          # Cleaned & feature-ready data
│
├── notebooks/
│   ├── 01-data-ingest.ipynb
│   └── 02_mumbai_aqi_eda.ipynb
│
├── src/
│   ├── data/               # Data preparation scripts
│   ├── features/           # Feature engineering
│   └── models/             # Model training & evaluation
│
├── deployments/
│   └── streamlit/
│       └── app.py          # Streamlit dashboard
│
├── models/                 # Trained model files
├── requirements.txt
└── README.md
---

## 🚀 Streamlit Application
The project is deployed as a **live Streamlit dashboard** that provides:
- Historical AQI visualization
- Next-day AQI prediction
- Health risk interpretation

📌 *Streamlit app link:*  
👉 **[(https://air-quality-forecasting-mumbai-t7kcwd9sa3w95ycycfr2mh.streamlit.app/)]**

---

## 🛠️ Technologies Used
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Streamlit
- Git & GitHub

---

## 📈 Key Insights
- Mumbai AQI shows strong **seasonal variation**
- PM2.5 is the most influential pollutant
- Lagged AQI values are the strongest predictors
- Several days fall under **Moderate to Poor** health risk categories

---

## 🔮 Future Enhancements
- Integrate weather data (temperature, humidity, wind)
- Extend forecasting to multi-day horizons
- Add real-time AQI alerts
- Explore deep learning models (LSTM)

---

## 👤 Author
**Prajakta Shinde**  
MSc Data Science  

---

## 📄 License
This project is intended for **academic and learning purposes**.
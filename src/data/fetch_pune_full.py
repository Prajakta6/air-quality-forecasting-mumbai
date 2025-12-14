import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
API_KEY = os.getenv("OPENAQ_API_KEY")

if not API_KEY:
    raise RuntimeError("OPENAQ_API_KEY not found. Check your .env file.")

HEADERS = {"X-API-Key": API_KEY}

BASE_LOC = "https://api.openaq.org/v3/locations"
BASE_MEAS = "https://api.openaq.org/v3/records"

# Pune coordinates
LAT, LON = 18.5204, 73.8567


def get_pune_locations():
    params = {
        "country": "IN",
        "city": "Pune",
        "limit": 50
    }

    r = requests.get(BASE_LOC, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()

    df = pd.json_normalize(data.get("results", []))
    if df.empty:
        print("❌ No Pune stations found")
        return []

    df.to_csv("data/raw/pune_locations_v3.csv", index=False)
    print("✅ Pune stations found:", len(df))
    return df["id"].tolist()


def fetch_measurements(location_id, parameter):
    print(f"Fetching {parameter} for location {location_id}")

    results = []
    page = 1

    # IMPORTANT: correct parameter mapping
    if parameter == "pm25":
        parameter = "pm2.5"

    while True:
        params = {
            "location": location_id,
            "parameter": parameter,
            "date_from": "2023-01-01",
            "date_to": "2024-12-31",
            "limit": 1000,
            "page": page
        }

        r = requests.get(BASE_MEAS, params=params, headers=HEADERS, timeout=30)

        if r.status_code == 404:
            print(f"No records endpoint data for location {location_id}")
            break

        r.raise_for_status()
        batch = r.json().get("results", [])

        if not batch:
            break

        results.extend(batch)
        page += 1
        time.sleep(0.3)

    if not results:
        return pd.DataFrame()

    df = pd.json_normalize(results)

    # v3 records already include datetime
    df["datetime"] = pd.to_datetime(df["datetime"])

    return df[["datetime", "parameter", "value", "unit", "location"]]


def main():
    print("🔍 Fetching Pune stations...")
    stations = get_pune_locations()

    pm25_all, pm10_all = [], []

    for sid in stations:
        pm25 = fetch_measurements(sid, "pm25")
        pm10 = fetch_measurements(sid, "pm10")

        if not pm25.empty:
            pm25_all.append(pm25)
        if not pm10.empty:
            pm10_all.append(pm10)

    if pm25_all:
        pd.concat(pm25_all).to_csv("data/raw/openaq_pune_pm25_full.csv", index=False)
        print("✅ Saved PM2.5 data")

    if pm10_all:
        pd.concat(pm10_all).to_csv("data/raw/openaq_pune_pm10_full.csv", index=False)
        print("✅ Saved PM10 data")


if __name__ == "__main__":
    main()
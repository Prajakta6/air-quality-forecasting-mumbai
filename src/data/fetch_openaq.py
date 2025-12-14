import requests
import pandas as pd
import time

BASE_URL = "https://api.openaq.org/v2/measurements"


def fetch_city_parameter(city, parameter, date_from, date_to, limit=10000):
    """
    Fetches AQI parameter data (e.g., pm25, pm10) for a given city from OpenAQ API.
    """
    results = []
    page = 1

    while True:
        params = {
            "city": city,
            "parameter": parameter,
            "date_from": date_from,
            "date_to": date_to,
            "limit": limit,
            "page": page
        }

        response = requests.get(BASE_URL, params=params, timeout=20)
        data = response.json()

        # Stop if no more results
        if "results" not in data or not data["results"]:
            break

        results.extend(data["results"])
        page += 1

        time.sleep(1)  # Avoid API rate limits

    # Convert results to DataFrame
    if not results:
        print(f"No data found for {city}, {parameter}")
        return pd.DataFrame()

    df = pd.json_normalize(results)

    # Keep the important columns only
    df["datetime"] = pd.to_datetime(df["date.local"])
    df = df[["datetime", "parameter", "value", "unit", "location"]]

    return df
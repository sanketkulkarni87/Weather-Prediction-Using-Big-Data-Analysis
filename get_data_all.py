import json
import time
import urllib.request
import urllib.parse
import urllib.error
import pandas as pd

CITIES = {
    "Delhi": (28.61, 77.21),
    "Mumbai": (19.08, 72.88),
    "Bengaluru": (12.97, 77.59),
    "Chennai": (13.08, 80.27),
    "Kolkata": (22.57, 88.36),
    "Hyderabad": (17.39, 78.49),
    "Pune": (18.52, 73.86),
    "Ahmedabad": (23.02, 72.57),
    "Jaipur": (26.91, 75.79),
    "Lucknow": (26.85, 80.95),
}

daily_vars = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "precipitation_sum", "rain_sum", "wind_speed_10m_max",
    "relative_humidity_2m_mean", "pressure_msl_mean", "cloud_cover_mean",
]

def fetch(city, lat, lon):
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": "2010-01-01", "end_date": "2025-12-31",
        "daily": ",".join(daily_vars), "timezone": "auto",
    }
    url = "https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(params)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = json.load(r)
            df = pd.DataFrame(data["daily"]).rename(columns={"time": "date"})
            df.insert(0, "city", city)
            return df
        except urllib.error.HTTPError as e:
            print(f"  {city}: HTTP {e.code}, retrying in 30s...")
            time.sleep(30)
    raise RuntimeError(f"Could not download {city}")

frames = []
for city, (lat, lon) in CITIES.items():
    print("Downloading", city)
    frames.append(fetch(city, lat, lon))
    time.sleep(5)

all_df = pd.concat(frames, ignore_index=True)
all_df.to_csv("weather_all.csv", index=False)
print("Saved weather_all.csv with shape", all_df.shape)
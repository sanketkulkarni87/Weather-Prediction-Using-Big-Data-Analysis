import json
import urllib.request
import urllib.parse
import pandas as pd

# Change these to your city (example values below)
CITY = "Delhi"
LAT, LON = 28.61, 77.21

daily_vars = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "precipitation_sum", "rain_sum", "wind_speed_10m_max",
    "relative_humidity_2m_mean", "pressure_msl_mean", "cloud_cover_mean",
]

params = {
    "latitude": LAT,
    "longitude": LON,
    "start_date": "2010-01-01",
    "end_date": "2025-12-31",
    "daily": ",".join(daily_vars),
    "timezone": "auto",
}

url = "https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(params)
with urllib.request.urlopen(url, timeout=60) as r:
    data = json.load(r)

df = pd.DataFrame(data["daily"]).rename(columns={"time": "date"})
df.insert(0, "city", CITY)
df.to_csv(f"weather_{CITY.lower()}.csv", index=False)

print(df.shape)
print(df.head())
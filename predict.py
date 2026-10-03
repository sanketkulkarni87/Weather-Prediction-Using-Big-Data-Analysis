import sys
import joblib
import pandas as pd

city = sys.argv[1] if len(sys.argv) > 1 else "Mumbai"
date = pd.Timestamp(sys.argv[2] if len(sys.argv) > 2 else "2025-07-15")

temp_model = joblib.load("models/temp_model.joblib")
rain_model = joblib.load("models/rain_model.joblib")
info = joblib.load("models/feature_info.joblib")

raw = pd.read_csv("weather_all.csv", parse_dates=["date"])
raw = raw[raw["city"] == city].sort_values("date").reset_index(drop=True)
if raw.empty:
    sys.exit(f"City '{city}' not found. Choose from: {sorted(pd.read_csv('weather_all.csv')['city'].unique())}")

def build_features(d):
    d = d.copy()
    d["month"] = d["date"].dt.month
    d["day_of_year"] = d["date"].dt.dayofyear
    for c in ["temperature_2m_mean", "relative_humidity_2m_mean", "pressure_msl_mean",
              "precipitation_sum", "wind_speed_10m_max", "cloud_cover_mean"]:
        d[f"{c}_lag1"] = d[c].shift(1)
    d["temperature_2m_mean_lag2"] = d["temperature_2m_mean"].shift(2)
    d["temp_avg_3d"] = d["temperature_2m_mean"].rolling(3, min_periods=1).mean()
    d["temp_avg_7d"] = d["temperature_2m_mean"].rolling(7, min_periods=1).mean()
    d["humidity_avg_3d"] = d["relative_humidity_2m_mean"].rolling(3, min_periods=1).mean()
    d["pressure_avg_3d"] = d["pressure_msl_mean"].rolling(3, min_periods=1).mean()
    d["rain_sum_7d"] = d["precipitation_sum"].rolling(7, min_periods=1).sum()
    d["pressure_change"] = d["pressure_msl_mean"] - d["pressure_msl_mean_lag1"]
    return d

feats = build_features(raw)
row = feats[feats["date"] == date]
if row.empty or row[info["num_cols"]].isna().any().any():
    sys.exit("Pick a date inside the data range, at least 3 days after 2010-01-01.")

X = row[info["num_cols"] + info["cat_cols"]]
pred_temp = temp_model.predict(X)[0]
rain_prob = rain_model.predict_proba(X)[0][1]

print(f"\nCity: {city} | Today: {date.date()}")
print(f"Today's mean temperature : {row['temperature_2m_mean'].iloc[0]:.1f} °C")
print(f"Predicted tomorrow temp  : {pred_temp:.1f} °C")
print(f"Rain tomorrow?           : {'YES' if rain_prob >= 0.5 else 'NO'} (model score {rain_prob:.2f})")

nxt = raw[raw["date"] == date + pd.Timedelta(days=1)]
if not nxt.empty:
    actual_t = nxt["temperature_2m_mean"].iloc[0]
    actual_r = nxt["precipitation_sum"].iloc[0]
    print(f"\nActual tomorrow          : {actual_t:.1f} °C, rainfall {actual_r:.1f} mm "
          f"({'rainy' if actual_r >= 1 else 'dry'})")
    
# Weather Prediction Using Big Data Analysis

A mini project that analyses 16 years of daily weather data for 10 Indian cities and predicts, one day ahead:

- **Tomorrow's average temperature** (regression)
- **Whether it will rain tomorrow** (classification, rain = 1 mm or more)

Data processing and feature engineering run on **Apache Spark (PySpark)**. Models are trained with both **scikit-learn** and **Spark MLlib**, and the results are shown in a **Streamlit** web app.

---

## Features

- Downloads historical weather data from the free Open-Meteo API (no API key needed)
- Spark pipeline: load, validate, aggregate, and build features with window functions
- Two prediction tasks, each compared against simple baselines
- Time-based train/test split (train 2010-2023, test 2024-2025) to avoid data leakage
- Same models built in scikit-learn and Spark MLlib for comparison
- Streamlit app with a weather-app style forecast, an all-cities view, and analytics

## Tech stack

Python, PySpark, Pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Joblib, Streamlit

## Dataset

| Item | Detail |
|---|---|
| Source | Open-Meteo Historical Weather API |
| Cities | Delhi, Mumbai, Bengaluru, Chennai, Kolkata, Hyderabad, Pune, Ahmedabad, Jaipur, Lucknow |
| Period | 2010-01-01 to 2025-12-31 |
| Size | 58,440 daily rows (10 cities x 5,844 days) |
| Variables | Max, min and mean temperature, precipitation, rain, max wind speed, mean humidity, mean pressure, mean cloud cover |

The data has no missing values or duplicates, so no cleaning was needed.

## Architecture

```
Open-Meteo API -> weather_all.csv -> Spark (checks, aggregations, window features)
                                          |
                                      features.csv
                                          |
                   scikit-learn models  +  Spark MLlib models
                                          |
                     evaluation vs baselines -> saved models -> Streamlit app
```

## Feature engineering

Spark window functions (partitioned by city, ordered by date) create:

- **Lag features:** yesterday's temperature, humidity, pressure, rainfall, wind and cloud cover, and the temperature two days ago
- **Rolling averages:** 3-day and 7-day temperature, 3-day humidity and pressure, 7-day rainfall total
- **Pressure change** compared with yesterday (falling pressure often comes before rain)
- **Calendar features:** month and day of the year
- **Targets:** tomorrow's mean temperature, and whether tomorrow's precipitation is 1 mm or more

## Project structure

```
weather_project/
├── get_data_all.py     # download data for 10 cities -> weather_all.csv
├── explore.py          # Spark: schema, row counts, missing values, duplicates, statistics
├── eda.py              # Spark aggregations + plots -> plots/
├── features.py         # Spark window-function features -> features.csv
├── step5_models.py     # scikit-learn training + evaluation -> models/, results.csv
├── mllib.py            # Spark MLlib training + evaluation -> results_spark.csv
├── predict.py          # command-line prediction demo
├── app.py              # Streamlit web app
├── weather_all.csv
├── features.csv
├── models/             # saved Random Forest models (joblib)
├── plots/              # charts used in the report and app
├── results.csv
└── results_spark.csv
```

Your script names may differ slightly; run them in the order shown under "How to run".

## Setup

1. Install Python 3.10 or newer and **Java 17 or newer** (needed by Spark).
2. Install the libraries:
   ```
   python -m pip install pyspark pandas numpy matplotlib seaborn scikit-learn joblib streamlit
   ```
3. Check that Spark starts:
   ```
   python -c "from pyspark.sql import SparkSession; s=SparkSession.builder.master('local[*]').getOrCreate(); print('Spark version:', s.version); s.stop()"
   ```

## How to run

Run these from inside the project folder, in this order:

```
python get_data_all.py      # 1. download data (takes a couple of minutes)
python explore.py           # 2. inspect and validate the data with Spark
python eda.py               # 3. aggregations and plots
python features.py          # 4. build features -> features.csv
python step5_models.py      # 5. train scikit-learn models
python mllib.py             # 6. train Spark MLlib models
python predict.py Delhi 2025-03-10    # 7. optional command-line demo
python -m streamlit run app.py        # 8. launch the web app
```

The app opens at http://localhost:8501. Press **Ctrl + C** in the terminal to stop it.

## Results

Models were tested on 2024-2025, which they never saw during training.

**Tomorrow's temperature** (lower error is better)

| Model | RMSE | MAE | R2 |
|---|---|---|---|
| Baseline (tomorrow = today) | 0.869 | 0.638 | 0.966 |
| Linear Regression | 0.824 | 0.609 | 0.969 |
| Random Forest (scikit-learn) | **0.818** | **0.605** | **0.970** |
| Linear Regression (MLlib) | 0.824 | 0.609 | 0.969 |
| Random Forest (MLlib) | 0.886 | 0.656 | 0.965 |

**Rain tomorrow** (higher is better)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Baseline (always dry) | 0.657 | 0.000 | 0.000 | 0.000 |
| Baseline (rain today -> rain tomorrow) | 0.855 | 0.789 | 0.789 | 0.789 |
| Logistic Regression | 0.857 | 0.752 | 0.869 | 0.806 |
| Random Forest (scikit-learn) | **0.872** | 0.773 | **0.887** | **0.826** |
| Logistic Regression (MLlib) | 0.857 | 0.752 | 0.869 | 0.806 |
| Random Forest (MLlib) | 0.869 | 0.769 | 0.883 | 0.822 |

## Key findings

- **Simple baselines are strong.** Temperature changes slowly, so "tomorrow = today" already gives R2 of 0.966. Random Forest improves on it, but only by a small margin (typical error about 0.6 °C).
- **Rain arrives in spells,** so "rain today -> rain tomorrow" already reaches 85.5% accuracy. Random Forest beats it, with better recall (it catches about 9 in 10 rainy days).
- **Linear models match across libraries.** Linear and Logistic Regression give identical scores in scikit-learn and MLlib, which confirms the pipeline is consistent.
- **MLlib Random Forest was weaker on temperature** than scikit-learn's, partly because of its depth limit and binning of feature values.
- **Spark was not faster at this size.** With about 58,000 rows on one laptop, Spark's overhead outweighs its benefit. Its advantage appears when data outgrows one machine, and this pipeline would carry over to a cluster.

## Limitations

- Uses daily summaries only, with no radar or satellite data
- Predicts one day ahead only
- The rain output is a model score, not a calibrated probability, because the models were trained with extra weight on rainy days
- Not an official forecast

## Future work

- Add hourly data and multi-day forecasts
- Try gradient boosting (XGBoost, LightGBM) or LSTM models
- Predict rainfall amount, not just rain or no rain
- Run Spark on a real cluster with data for many more locations
- Deploy the app online

## Notes on Windows warnings

Messages such as `Did not find winutils.exe`, `Unable to load native-hadoop library`, and `netlib-blas ... openblas.dll failed` are harmless for local Spark and can be ignored.

## Data source

Weather data from [Open-Meteo](https://open-meteo.com/) (Historical Weather API).
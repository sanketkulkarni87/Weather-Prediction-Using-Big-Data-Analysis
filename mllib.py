import time
import pandas as pd
from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator

spark = (SparkSession.builder.master("local[*]").appName("WeatherMLlib")
         .config("spark.driver.memory", "4g")
         .config("spark.sql.shuffle.partitions", "8")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

df = spark.read.csv("features.csv", header=True, inferSchema=True)
df = df.withColumn("rain_tomorrow", F.col("rain_tomorrow").cast("double"))

exclude = {"city", "date", "year", "rain_sum", "temp_tomorrow", "rain_tomorrow"}
num_cols = [c for c in df.columns if c not in exclude]

def make_prep():
    return [
        StringIndexer(inputCol="city", outputCol="city_idx", handleInvalid="keep"),
        OneHotEncoder(inputCols=["city_idx"], outputCols=["city_vec"]),
        VectorAssembler(inputCols=num_cols + ["city_vec"], outputCol="raw"),
        StandardScaler(inputCol="raw", outputCol="features", withMean=False, withStd=True),
    ]

# Same time-based split as before
train = df.filter(F.col("year") <= 2023).cache()
test = df.filter(F.col("year") >= 2024).cache()
print("Train rows:", train.count(), "| Test rows:", test.count())

results = []

# ---------------- Temperature (regression) ----------------
def run_reg(name, est):
    t0 = time.time()
    pred = Pipeline(stages=make_prep() + [est]).fit(train).transform(test)
    row = {"task": "temperature", "model": name}
    for metric in ["rmse", "mae", "r2"]:
        ev = RegressionEvaluator(labelCol="temp_tomorrow", predictionCol="prediction",
                                 metricName=metric)
        row[metric.upper()] = round(ev.evaluate(pred), 3)
    row["seconds"] = round(time.time() - t0, 1)
    results.append(row)

run_reg("MLlib Linear Regression",
        LinearRegression(featuresCol="features", labelCol="temp_tomorrow"))
run_reg("MLlib Random Forest",
        RandomForestRegressor(featuresCol="features", labelCol="temp_tomorrow",
                              numTrees=100, maxDepth=10, seed=42))

# ---------------- Rain (classification) ----------------
pos = train.filter(F.col("rain_tomorrow") == 1.0).count()
neg = train.count() - pos
train_c = train.withColumn("w", F.when(F.col("rain_tomorrow") == 1.0, neg / pos).otherwise(1.0))

def run_clf(name, est):
    t0 = time.time()
    pred = Pipeline(stages=make_prep() + [est]).fit(train_c).transform(test)
    row = {"task": "rain", "model": name}
    for label, metric in [("Accuracy", "accuracy"), ("Precision", "precisionByLabel"),
                          ("Recall", "recallByLabel"), ("F1", "fMeasureByLabel")]:
        ev = MulticlassClassificationEvaluator(labelCol="rain_tomorrow",
                                               predictionCol="prediction",
                                               metricName=metric, metricLabel=1.0)
        row[label] = round(ev.evaluate(pred), 3)
    row["seconds"] = round(time.time() - t0, 1)
    results.append(row)

run_clf("MLlib Logistic Regression",
        LogisticRegression(featuresCol="features", labelCol="rain_tomorrow",
                           weightCol="w", maxIter=100))
run_clf("MLlib Random Forest",
        RandomForestClassifier(featuresCol="features", labelCol="rain_tomorrow",
                               weightCol="w", numTrees=100, maxDepth=10, seed=42))

# ---------------- Print + save ----------------
res = pd.DataFrame(results)
print("\n=== TEMPERATURE (Spark MLlib) ===")
print(res[res.task == "temperature"].dropna(axis=1, how="all").drop(columns="task").to_string(index=False))
print("\n=== RAIN (Spark MLlib) ===")
print(res[res.task == "rain"].dropna(axis=1, how="all").drop(columns="task").to_string(index=False))
res.to_csv("results_spark.csv", index=False)

spark.stop()
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window

spark = (SparkSession.builder.master("local[*]")
         .appName("WeatherFeatures").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

df = spark.read.csv("weather_all.csv", header=True, inferSchema=True)

w = Window.partitionBy("city").orderBy("date")
w3 = w.rowsBetween(-2, 0)    # today + previous 2 days
w7 = w.rowsBetween(-6, 0)    # today + previous 6 days

df = (df
      .withColumn("year", F.year("date"))
      .withColumn("month", F.month("date"))
      .withColumn("day_of_year", F.dayofyear("date")))

# Lag features (yesterday and the day before)
for c in ["temperature_2m_mean", "relative_humidity_2m_mean",
          "pressure_msl_mean", "precipitation_sum", "wind_speed_10m_max",
          "cloud_cover_mean"]:
    df = df.withColumn(f"{c}_lag1", F.lag(c, 1).over(w))
df = df.withColumn("temperature_2m_mean_lag2", F.lag("temperature_2m_mean", 2).over(w))

# Rolling averages / sums
df = (df
      .withColumn("temp_avg_3d", F.avg("temperature_2m_mean").over(w3))
      .withColumn("temp_avg_7d", F.avg("temperature_2m_mean").over(w7))
      .withColumn("humidity_avg_3d", F.avg("relative_humidity_2m_mean").over(w3))
      .withColumn("pressure_avg_3d", F.avg("pressure_msl_mean").over(w3))
      .withColumn("rain_sum_7d", F.sum("precipitation_sum").over(w7))
      .withColumn("pressure_change", F.col("pressure_msl_mean") - F.col("pressure_msl_mean_lag1")))

# Targets: tomorrow's temperature and tomorrow's rain (>= 1 mm)
df = (df
      .withColumn("temp_tomorrow", F.lead("temperature_2m_mean", 1).over(w))
      .withColumn("rain_tomorrow",
                  (F.lead("precipitation_sum", 1).over(w) >= 1).cast("int")))

# Drop rows where lags/targets don't exist (first days and last day of each city)
df = df.dropna()

print("Rows after feature engineering:", df.count())
print("\n=== Tomorrow rain class balance ===")
df.groupBy("rain_tomorrow").count().show()
print("=== Rows per year ===")
df.groupBy("year").count().orderBy("year").show(20)

df.printSchema()

# Save for the modelling step
df.orderBy("city", "date").toPandas().to_csv("features.csv", index=False)
print("Saved features.csv")
spark.stop()
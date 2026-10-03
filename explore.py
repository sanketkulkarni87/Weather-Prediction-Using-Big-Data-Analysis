from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (SparkSession.builder
         .master("local[*]")
         .appName("WeatherPrediction")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

df = spark.read.csv("weather_all.csv", header=True, inferSchema=True)

print("\n=== Schema ===")
df.printSchema()

print("=== Row count ===")
print(df.count())

print("\n=== Sample rows ===")
df.show(5)

print("=== Rows per city ===")
df.groupBy("city").count().orderBy("city").show()

print("=== Missing values per column ===")
df.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns]).show()

print("=== Duplicate rows (city + date) ===")
print(df.count() - df.dropDuplicates(["city", "date"]).count())

print("=== Basic statistics ===")
df.select("temperature_2m_mean", "precipitation_sum",
          "relative_humidity_2m_mean", "wind_speed_10m_max").describe().show()

spark.stop()
import os
from pyspark.sql import SparkSession, functions as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

spark = (SparkSession.builder.master("local[*]")
         .appName("WeatherEDA").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
os.makedirs("plots", exist_ok=True)

df = spark.read.csv("weather_all.csv", header=True, inferSchema=True)
df = (df.withColumn("year", F.year("date"))
        .withColumn("month", F.month("date"))
        .withColumn("rainy", (F.col("precipitation_sum") >= 1).cast("int")))

# 1. City-wise summary (Spark aggregation)
print("=== City summary ===")
city_stats = df.groupBy("city").agg(
    F.round(F.avg("temperature_2m_mean"), 1).alias("avg_temp_C"),
    F.round(F.avg("relative_humidity_2m_mean"), 1).alias("avg_humidity"),
    F.round(F.sum("precipitation_sum") / 16, 0).alias("avg_annual_rain_mm"),
    F.round(F.avg("rainy") * 100, 1).alias("rainy_days_pct"),
)
city_stats.orderBy("city").show()

# 2. Overall rain class balance
print("=== Rainy vs dry days (rain >= 1 mm) ===")
df.groupBy("rainy").count().show()

# 3. Monthly average temperature and rainfall per city
monthly = (df.groupBy("city", "month")
             .agg(F.avg("temperature_2m_mean").alias("temp"),
                  F.avg("precipitation_sum").alias("rain"))
             .orderBy("city", "month").toPandas())

plt.figure(figsize=(10, 5))
sns.lineplot(data=monthly, x="month", y="temp", hue="city", marker="o")
plt.title("Average temperature by month")
plt.ylabel("Temperature (°C)")
plt.xticks(range(1, 13))
plt.tight_layout()
plt.savefig("plots/monthly_temp.png", dpi=120)
plt.close()

plt.figure(figsize=(10, 5))
sns.lineplot(data=monthly, x="month", y="rain", hue="city", marker="o")
plt.title("Average daily rainfall by month")
plt.ylabel("Rainfall (mm/day)")
plt.xticks(range(1, 13))
plt.tight_layout()
plt.savefig("plots/monthly_rain.png", dpi=120)
plt.close()

# 4. Yearly temperature trend
yearly = (df.groupBy("year").agg(F.avg("temperature_2m_mean").alias("temp"))
            .orderBy("year").toPandas())
plt.figure(figsize=(8, 4))
sns.lineplot(data=yearly, x="year", y="temp", marker="o")
plt.title("Average temperature by year (all cities)")
plt.ylabel("Temperature (°C)")
plt.tight_layout()
plt.savefig("plots/yearly_temp.png", dpi=120)
plt.close()

# 5. Correlation heatmap
num_cols = ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
            "precipitation_sum", "wind_speed_10m_max",
            "relative_humidity_2m_mean", "pressure_msl_mean", "cloud_cover_mean"]
pdf = df.select(num_cols).toPandas()
plt.figure(figsize=(9, 7))
sns.heatmap(pdf.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation between weather variables")
plt.tight_layout()
plt.savefig("plots/correlation.png", dpi=120)
plt.close()

print("Plots saved in the 'plots' folder.")
spark.stop()
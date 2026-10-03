import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

os.makedirs("plots", exist_ok=True)
os.makedirs("models", exist_ok=True)

df = pd.read_csv("features.csv", parse_dates=["date"])

# ---------- Features / split ----------
targets = ["temp_tomorrow", "rain_tomorrow"]
drop_cols = targets + ["date", "year", "rain_sum"]      # year: avoid trend leakage; rain_sum ~ duplicate
num_cols = [c for c in df.columns if c not in drop_cols + ["city"]]
cat_cols = ["city"]

train = df[df["year"] <= 2023]
test = df[df["year"] >= 2024]
X_train, X_test = train[num_cols + cat_cols], test[num_cols + cat_cols]
print(f"Train rows: {len(train)}  |  Test rows: {len(test)}")

def make_pre():
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])

results = []

# =====================================================
# PART A: Temperature (regression)
# =====================================================
y_tr, y_te = train["temp_tomorrow"], test["temp_tomorrow"]

def reg_metrics(name, pred):
    rmse = np.sqrt(mean_squared_error(y_te, pred))
    results.append({"task": "temperature", "model": name,
                    "RMSE": round(rmse, 3),
                    "MAE": round(mean_absolute_error(y_te, pred), 3),
                    "R2": round(r2_score(y_te, pred), 3)})

reg_metrics("Baseline (tomorrow = today)", test["temperature_2m_mean"])

lin = Pipeline([("pre", make_pre()), ("model", LinearRegression())])
lin.fit(X_train, y_tr)
reg_metrics("Linear Regression", lin.predict(X_test))

rf_reg = Pipeline([("pre", make_pre()),
                   ("model", RandomForestRegressor(n_estimators=200, max_depth=15,
                                                   min_samples_leaf=3, n_jobs=-1,
                                                   random_state=42))])
rf_reg.fit(X_train, y_tr)
rf_pred = rf_reg.predict(X_test)
reg_metrics("Random Forest", rf_pred)

# =====================================================
# PART B: Rain tomorrow (classification)
# =====================================================
yc_tr, yc_te = train["rain_tomorrow"], test["rain_tomorrow"]

def clf_metrics(name, pred):
    results.append({"task": "rain", "model": name,
                    "Accuracy": round(accuracy_score(yc_te, pred), 3),
                    "Precision": round(precision_score(yc_te, pred, zero_division=0), 3),
                    "Recall": round(recall_score(yc_te, pred), 3),
                    "F1": round(f1_score(yc_te, pred), 3)})

clf_metrics("Baseline (always dry)", np.zeros(len(test), dtype=int))
clf_metrics("Baseline (rain today -> rain tomorrow)", (test["precipitation_sum"] >= 1).astype(int))

log = Pipeline([("pre", make_pre()),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))])
log.fit(X_train, yc_tr)
clf_metrics("Logistic Regression", log.predict(X_test))

rf_clf = Pipeline([("pre", make_pre()),
                   ("model", RandomForestClassifier(n_estimators=200, max_depth=12,
                                                    min_samples_leaf=5, n_jobs=-1,
                                                    class_weight="balanced",
                                                    random_state=42))])
rf_clf.fit(X_train, yc_tr)
rf_c_pred = rf_clf.predict(X_test)
clf_metrics("Random Forest", rf_c_pred)

# ---------- Print + save results ----------
res = pd.DataFrame(results)
print("\n=== TEMPERATURE RESULTS (lower RMSE/MAE is better) ===")
print(res[res.task == "temperature"].dropna(axis=1, how="all").drop(columns="task").to_string(index=False))
print("\n=== RAIN RESULTS ===")
print(res[res.task == "rain"].dropna(axis=1, how="all").drop(columns="task").to_string(index=False))
res.to_csv("results.csv", index=False)

# ---------- Plots ----------
# 1. Actual vs predicted temperature
samp = np.random.RandomState(0).choice(len(y_te), 2000, replace=False)
plt.figure(figsize=(6, 6))
plt.scatter(y_te.values[samp], rf_pred[samp], s=6, alpha=0.4)
lo, hi = y_te.min(), y_te.max()
plt.plot([lo, hi], [lo, hi], "r--")
plt.xlabel("Actual temperature (°C)")
plt.ylabel("Predicted temperature (°C)")
plt.title("Random Forest: actual vs predicted (test set)")
plt.tight_layout(); plt.savefig("plots/temp_actual_vs_pred.png", dpi=120); plt.close()

# 2. Confusion matrix (rain)
cm = confusion_matrix(yc_te, rf_c_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Dry", "Rain"], yticklabels=["Dry", "Rain"])
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.title("Rain prediction: Random Forest")
plt.tight_layout(); plt.savefig("plots/rain_confusion_matrix.png", dpi=120); plt.close()

# 3. Feature importance
def plot_importance(pipe, title, fname):
    names = pipe.named_steps["pre"].get_feature_names_out()
    imp = pd.Series(pipe.named_steps["model"].feature_importances_, index=names)
    imp = imp.sort_values().tail(10)
    plt.figure(figsize=(8, 5))
    imp.plot(kind="barh")
    plt.title(title)
    plt.tight_layout(); plt.savefig(fname, dpi=120); plt.close()

plot_importance(rf_reg, "Top 10 features: temperature model", "plots/importance_temp.png")
plot_importance(rf_clf, "Top 10 features: rain model", "plots/importance_rain.png")

# ---------- Save models ----------
joblib.dump(rf_reg, "models/temp_model.joblib")
joblib.dump(rf_clf, "models/rain_model.joblib")
joblib.dump({"num_cols": num_cols, "cat_cols": cat_cols}, "models/feature_info.joblib")
print("\nSaved models in 'models' and plots in 'plots'.")
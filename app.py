"""SkyCast India - weather prediction app. Run: python -m streamlit run app.py"""
import os
from datetime import date, timedelta
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SkyCast India", page_icon="🌦️", layout="wide")
B = os.path.dirname(os.path.abspath(__file__))
P = lambda *a: os.path.join(B, *a)

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
html, body, .stApp {font-family:'Outfit',sans-serif;}
.stApp {background:linear-gradient(160deg,#0b1d3a 0%,#12395f 55%,#1d5c85 100%);}
[data-testid="stHeader"] {background:transparent;}
[data-testid="stSidebar"] {background:#081527;}
.stApp h1,.stApp h2,.stApp h3,.stApp [data-testid="stMarkdownContainer"] p,.stApp label p,
.stApp [data-testid="stCaptionContainer"] {color:#eaf3ff;}
.brand {font-size:2.6rem;font-weight:700;color:#fff;margin:0;letter-spacing:-.5px}
.sub {color:#a9c4e4;margin:0 0 18px 0}
.card {border-radius:22px;padding:26px 30px;color:#fff;box-shadow:0 12px 32px rgba(0,0,0,.28);min-height:250px}
.sunny {background:linear-gradient(135deg,#f9b233,#e2701f)}
.cloudy {background:linear-gradient(135deg,#8497b0,#4a6484)}
.rain {background:linear-gradient(135deg,#3e6b94,#0d3b66)}
.lbl {font-size:.95rem;opacity:.85;font-weight:400}
.big {font-size:5rem;font-weight:300;line-height:1.05}
.cnd {font-size:1.35rem;font-weight:600}
.foot {margin-top:12px;font-size:.95rem;opacity:.9}
.pill {display:inline-block;background:rgba(255,255,255,.22);border-radius:99px;padding:3px 12px;font-size:.8rem;margin-left:8px}
.tiles {display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:16px 0}
.tile {background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.15);border-radius:16px;padding:14px 16px;color:#cfe2f7;font-size:.9rem}
.tile b {display:block;font-size:1.5rem;font-weight:600;color:#fff}
.mini {border-radius:16px;padding:14px;color:#fff;text-align:center;margin-bottom:12px}
.mini .t {font-size:1.9rem;font-weight:600}
</style>""", unsafe_allow_html=True)

need = [P("weather_all.csv")] + [P("models", f"{m}.joblib") for m in ("temp_model", "rain_model", "feature_info")]
if any(not os.path.exists(f) for f in need):
    st.error("Put app.py inside your weather_project folder (with weather_all.csv and models/).")
    st.stop()


@st.cache_resource
def load_models():
    return [joblib.load(P("models", f"{m}.joblib")) for m in ("temp_model", "rain_model", "feature_info")]


@st.cache_data
def load_raw():
    return pd.read_csv(P("weather_all.csv"), parse_dates=["date"]).sort_values(["city", "date"]).reset_index(drop=True)


@st.cache_data
def feats(city):
    d = load_raw()
    d = d[d.city == city].reset_index(drop=True).copy()
    d["month"], d["day_of_year"] = d.date.dt.month, d.date.dt.dayofyear
    for c in ["temperature_2m_mean", "relative_humidity_2m_mean", "pressure_msl_mean",
              "precipitation_sum", "wind_speed_10m_max", "cloud_cover_mean"]:
        d[f"{c}_lag1"] = d[c].shift(1)
    d["temperature_2m_mean_lag2"] = d.temperature_2m_mean.shift(2)
    d["temp_avg_3d"] = d.temperature_2m_mean.rolling(3, min_periods=1).mean()
    d["temp_avg_7d"] = d.temperature_2m_mean.rolling(7, min_periods=1).mean()
    d["humidity_avg_3d"] = d.relative_humidity_2m_mean.rolling(3, min_periods=1).mean()
    d["pressure_avg_3d"] = d.pressure_msl_mean.rolling(3, min_periods=1).mean()
    d["rain_sum_7d"] = d.precipitation_sum.rolling(7, min_periods=1).sum()
    d["pressure_change"] = d.pressure_msl_mean - d.pressure_msl_mean_lag1
    return d


tm, rm, info = load_models()
NUM, CAT = info["num_cols"], info["cat_cols"]
raw = load_raw()
cities = sorted(raw.city.unique())
first, last = raw.date.min().date() + timedelta(days=2), raw.date.max().date()


@st.cache_data
def predict(city, d):
    f = feats(city)
    r = f[f.date == pd.Timestamp(d)]
    if r.empty or r[NUM].isna().any().any():
        return None
    X = r[NUM + CAT]
    return {"temp": float(tm.predict(X)[0]), "score": float(rm.predict_proba(X)[0][1]), "row": r.iloc[0]}


def cond(cloud, rain):
    if rain: return "🌧️", "Rain", "rain"
    if cloud >= 70: return "☁️", "Cloudy", "cloudy"
    if cloud >= 35: return "⛅", "Partly cloudy", "cloudy"
    return "☀️", "Clear sky", "sunny"


st.sidebar.markdown("### Location and date")
city = st.sidebar.selectbox("City", cities, index=cities.index("Mumbai") if "Mumbai" in cities else 0)
default = date(2025, 7, 15) if first <= date(2025, 7, 15) <= last else last
day = st.sidebar.date_input("Today's date", default, min_value=first, max_value=last)
thr = st.sidebar.slider("Rain alert threshold", 0.1, 0.9, 0.5, 0.05,
                        help="Rain is shown when the model's rain score reaches this value.")
tomorrow = day + timedelta(days=1)

st.markdown('<p class="brand">🌦️ SkyCast India</p><p class="sub">Next-day weather prediction from 16 years of data across 10 cities</p>',
            unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["Forecast", "India at a glance", "Analytics"])

with tab1:
    res = predict(city, day)
    if res is None:
        st.warning("Not enough history for this date. Pick a later date.")
    else:
        r = res["row"]
        rain_now = r.precipitation_sum >= 1
        e1, n1, c1 = cond(r.cloud_cover_mean, rain_now)
        rain_next = res["score"] >= thr
        e2, n2, c2 = cond(r.cloud_cover_mean, rain_next)
        badge = "Unseen test date" if day >= date(2024, 1, 1) else "Training date"
        nxt = raw[(raw.city == city) & (raw.date == pd.Timestamp(tomorrow))]
        actual = (f"Actual: {nxt.temperature_2m_mean.iloc[0]:.1f}°C, {nxt.precipitation_sum.iloc[0]:.1f} mm rain"
                  if not nxt.empty else "No actual data for this day")
        a, b = st.columns(2)
        a.markdown(f'<div class="card {c1}"><div class="lbl">{city} · {day:%a, %d %b %Y}<span class="pill">Observed</span></div>'
                   f'<div class="big">{r.temperature_2m_mean:.0f}°</div><div class="cnd">{e1} {n1}</div>'
                   f'<div class="foot">High {r.temperature_2m_max:.0f}° · Low {r.temperature_2m_min:.0f}°</div></div>',
                   unsafe_allow_html=True)
        b.markdown(f'<div class="card {c2}"><div class="lbl">Tomorrow · {tomorrow:%a, %d %b}<span class="pill">Predicted</span>'
                   f'<span class="pill">{badge}</span></div><div class="big">{res["temp"]:.0f}°</div>'
                   f'<div class="cnd">{e2} {n2}</div><div class="foot">Rain score {res["score"]:.2f} · {actual}</div></div>',
                   unsafe_allow_html=True)
        tiles = [("💧 Humidity", f"{r.relative_humidity_2m_mean:.0f}%"), ("💨 Wind (max)", f"{r.wind_speed_10m_max:.0f} km/h"),
                 ("🧭 Pressure", f"{r.pressure_msl_mean:.0f} hPa"), ("☁️ Cloud cover", f"{r.cloud_cover_mean:.0f}%"),
                 ("🌧️ Rain, 7 days", f"{r.rain_sum_7d:.0f} mm"), ("📉 Pressure change", f"{r.pressure_change:+.1f} hPa")]
        st.markdown('<div class="tiles">' + "".join(f'<div class="tile">{k}<b>{v}</b></div>' for k, v in tiles) + "</div>",
                    unsafe_allow_html=True)
        h = feats(city)
        h = h[(h.date <= pd.Timestamp(day)) & (h.date > pd.Timestamp(day) - pd.Timedelta(days=14))]
        fig, ax = plt.subplots(figsize=(10, 3.2))
        ax.plot(h.date, h.temperature_2m_mean, color="#7fd1ff", lw=2.5, marker="o", ms=4)
        ax.scatter([pd.Timestamp(tomorrow)], [res["temp"]], color="#ffb84d", s=110, zorder=3, label="Predicted tomorrow")
        if not nxt.empty:
            ax.scatter([pd.Timestamp(tomorrow)], [nxt.temperature_2m_mean.iloc[0]], color="#6ef3b0", s=90, marker="D", zorder=3, label="Actual")
        fig.patch.set_alpha(0); ax.set_facecolor("none")
        for s in ax.spines.values(): s.set_visible(False)
        ax.tick_params(colors="#cfe2f7"); ax.grid(alpha=.15); ax.set_ylabel("°C", color="#cfe2f7")
        ax.set_title("Last 14 days", color="#eaf3ff", loc="left")
        ax.legend(frameon=False, labelcolor="#eaf3ff"); fig.autofmt_xdate(); fig.tight_layout()
        st.pyplot(fig); plt.close(fig)
        st.caption("Rain score is a model score, not a true probability. Rain means 1 mm or more in a day. Learning project, not an official forecast.")

with tab2:
    st.subheader(f"Predicted weather for {tomorrow:%A, %d %b %Y}")
    cols = st.columns(5)
    for i, c in enumerate(cities):
        p = predict(c, day)
        if p is None:
            continue
        e, n, k = cond(p["row"].cloud_cover_mean, p["score"] >= thr)
        cols[i % 5].markdown(f'<div class="mini {k}"><div>{c}</div><div style="font-size:2rem">{e}</div>'
                             f'<div class="t">{p["temp"]:.0f}°</div><div style="font-size:.85rem">{n} · score {p["score"]:.2f}</div></div>',
                             unsafe_allow_html=True)

with tab3:
    yrs = raw.date.dt.year.nunique()
    st.subheader("City climate summary")
    st.dataframe(raw.assign(rainy=(raw.precipitation_sum >= 1) * 1).groupby("city").agg(
        avg_temp_C=("temperature_2m_mean", "mean"), avg_humidity=("relative_humidity_2m_mean", "mean"),
        annual_rain_mm=("precipitation_sum", lambda s: s.sum() / yrs), rainy_days_pct=("rainy", lambda s: s.mean() * 100)).round(1))
    st.subheader("Model results (tested on 2024-2025)")
    l, r_ = st.columns(2)
    for col, f, t in [(l, "results.csv", "scikit-learn"), (r_, "results_spark.csv", "Spark MLlib")]:
        if os.path.exists(P(f)):
            df = pd.read_csv(P(f)); col.markdown(f"**{t}**")
            for task in ("temperature", "rain"):
                col.dataframe(df[df.task == task].dropna(axis=1, how="all").drop(columns="task"), hide_index=True)
    imgs = [f for f in ("temp_actual_vs_pred.png", "rain_confusion_matrix.png", "importance_temp.png", "importance_rain.png")
            if os.path.exists(P("plots", f))]
    for i in range(0, len(imgs), 2):
        for col, f in zip(st.columns(2), imgs[i:i + 2]):
            col.image(P("plots", f))

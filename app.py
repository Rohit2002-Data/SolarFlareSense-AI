# app.py

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from fetch_donki import get_flare_data
import time

# UI setup
st.set_page_config(page_title="SolarFlareSense AI", page_icon="☀️")
st.title("☀️ SolarFlareSense AI — NASA Solar Flare Intensity Predictor")
st.markdown("Predict whether a solar flare will be C, M, or X class using NASA's real-time DONKI data.")

# 🔁 Auto-refresh logic
refresh_interval = st.sidebar.selectbox("🔁 Auto-refresh interval (minutes):", [0, 1, 5, 10], index=2)
if refresh_interval > 0:
    st.sidebar.info(f"⏳ Auto-refreshing every {refresh_interval} minute(s).")
    if "last_refresh" not in st.session_state:
        st.session_state["last_refresh"] = time.time()
    if time.time() - st.session_state["last_refresh"] > refresh_interval * 60:
        st.session_state["last_refresh"] = time.time()
        st.experimental_rerun()

# ⏱️ Days back selector
days = st.sidebar.slider("📅 Days of past data to fetch", 30, 365, 180)

# 🔭 Load live data
df = get_flare_data(days_back=days)
st.success(f"✅ Loaded {len(df)} solar flare events from NASA DONKI")

# 🧪 Feature engineering
df["latitude"] = df["sourceLocation"].str.extract(r'([NS]\d+)')[0]
df["longitude"] = df["sourceLocation"].str.extract(r'([EW]\d+)')[0]

def convert_coord(coord):
    if pd.isna(coord): return None
    return float(coord[1:]) * (-1 if coord[0] in ['S', 'W'] else 1)

df["latitude"] = df["latitude"].apply(convert_coord)
df["longitude"] = df["longitude"].apply(convert_coord)

features = ["activeRegionNum", "duration_minutes", "latitude", "longitude"]
df = df.dropna(subset=features + ["classType"])

X = df[features]
y = df["classType"]
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=42)

# 🧠 Train model
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# 📊 Evaluation
st.markdown("### 📈 Model Evaluation")
y_pred = model.predict(X_test)
acc = (y_pred == y_test).mean()
st.markdown(f"**✅ Accuracy:** `{acc*100:.2f}%`")

with st.expander("🧾 Confusion Matrix"):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
                xticklabels=le.classes_,
                yticklabels=le.classes_)
    st.pyplot(fig)

# 🔮 Prediction UI
st.header("🔮 Predict Solar Flare Class")

mode = st.radio("Select Input Method:", ["Manual Input", "From Live NASA Data"])

if mode == "Manual Input":
    st.subheader("🧪 Enter Flare Details")
    ar = st.number_input("Active Region Number", min_value=0, value=12987)
    dur = st.number_input("Duration (minutes)", min_value=1.0, value=20.0)
    lat = st.number_input("Latitude (positive N / negative S)", value=15.0)
    lon = st.number_input("Longitude (positive E / negative W)", value=30.0)

    if st.button("🔍 Predict Flare Class"):
        pred = model.predict([[ar, dur, lat, lon]])[0]
        label = le.inverse_transform([pred])[0]
        st.success(f"☀️ Predicted Flare Class: `{label}`")

else:
    st.subheader("📡 Choose Flare from NASA")
    selected = st.selectbox("Select event time", df["peakTime"])
    row = df[df["peakTime"] == selected]
    input_vals = row[features].values[0]

    st.write("📥 Input features:")
    st.write(dict(zip(features, input_vals)))

    if st.button("🔍 Predict Selected Flare"):
        pred = model.predict([input_vals])[0]
        st.success(f"☀️ Predicted Flare Class: `{le.inverse_transform([pred])[0]}`")


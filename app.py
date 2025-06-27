import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from fetch_donki import get_flare_data

# Streamlit config
st.set_page_config(page_title="SolarFlareSense AI", page_icon="☀️")
st.title("☀️ SolarFlareSense AI — Real-Time Flare Class Predictor")
st.markdown("Predict whether a solar flare will be **C**, **M**, or **X** class using NASA DONKI real-time data.")

# Sidebar: date range
days = st.sidebar.slider("📆 How many days back to fetch flare data?", 30, 365, 180)
df = get_flare_data(days_back=days)

# 🧪 Feature Engineering
df["duration_minutes"] = pd.to_numeric(df["duration_minutes"], errors="coerce")

def extract_coord(val, direction):
    if pd.isna(val): return None
    try:
        import re
        match = re.findall(rf"{direction}(\d+)", val)
        return float(match[0]) * (-1 if direction in ['S', 'W'] else 1) if match else None
    except: return None

df["latitude"] = df["sourceLocation"].apply(lambda x: extract_coord(x, "N") or extract_coord(x, "S"))
df["longitude"] = df["sourceLocation"].apply(lambda x: extract_coord(x, "E") or extract_coord(x, "W"))

# Define features
features = ["activeRegionNum", "duration_minutes", "latitude", "longitude"]

# Drop rows with missing label
df = df.dropna(subset=["classType"])

# Fill missing feature values with -1
df[features] = df[features].fillna(-1)

# Safety check
if df.empty:
    st.error("❌ No valid flare records found. Try expanding the date range.")
    st.stop()

# ML modeling
X = df[features]
y = df["classType"]
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=42)
model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

# Evaluation
st.markdown("### 📊 Model Evaluation")
acc = model.score(X_test, y_test)
st.write(f"✅ **Accuracy on test set:** `{acc * 100:.2f}%`")

with st.expander("📋 Classification Report"):
    y_pred = model.predict(X_test)

    report = classification_report(
        y_test,
        y_pred,
        labels=range(len(le.classes_)),      # ✅ Fixed mismatch
        target_names=le.classes_,
        output_dict=True
    )

    df_report = pd.DataFrame(report).transpose()
    st.dataframe(df_report.style.format({
        "precision": "{:.2f}",
        "recall": "{:.2f}",
        "f1-score": "{:.2f}",
        "support": "{:.0f}"
    }))

# 🔮 Prediction
st.markdown("### 🔮 Predict Solar Flare Class")
mode = st.radio("Choose Input Mode:", ["Manual Entry", "Select NASA Event"])

if mode == "Manual Entry":
    ar = st.number_input("Active Region Number", min_value=0, value=13000)
    dur = st.number_input("Duration (minutes)", min_value=0.0, value=30.0)
    lat = st.number_input("Latitude (N=+, S=-)", value=10.0)
    lon = st.number_input("Longitude (E=+, W=-)", value=20.0)

    if st.button("🚀 Predict Class"):
        pred = model.predict([[ar, dur, lat, lon]])[0]
        st.success(f"☀️ Predicted Flare Class: `{le.inverse_transform([pred])[0]}`")

else:
    selected = st.selectbox("Choose peak time", df["peakTime"])
    row = df[df["peakTime"] == selected]
    inputs = row[features].values[0]
    st.json(dict(zip(features, inputs)))

    if st.button("🚀 Predict from NASA Record"):
        pred = model.predict([inputs])[0]
        st.success(f"☀️ Predicted Flare Class: `{le.inverse_transform([pred])[0]}`")

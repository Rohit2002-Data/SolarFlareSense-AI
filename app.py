# app.py

import streamlit as st
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from fetch_donki import get_flare_data

# Streamlit UI setup
st.set_page_config(page_title="SolarFlareSense AI", page_icon="☀️")
st.title("☀️ SolarFlareSense AI — Real-Time Flare Class Predictor")
st.markdown("Predict whether a solar flare will be **C**, **M**, or **X** class using NASA DONKI real-time data.")

# 🔁 Auto-refresh selector
refresh_interval = st.sidebar.selectbox("🔁 Auto-refresh interval (minutes):", [0, 1, 5, 10], index=1)
if refresh_interval > 0:
    st.sidebar.info(f"⏳ Auto-refreshing every {refresh_interval} min.")
    if "last_refresh" not in st.session_state:
        st.session_state["last_refresh"] = time.time()
    if time.time() - st.session_state["last_refresh"] > refresh_interval * 60:
        st.session_state["last_refresh"] = time.time()
        st.rerun()

# 📅 Date range selector
days = st.sidebar.slider("📆 How many days back to fetch NASA data?", 30, 365, 180)
df = get_flare_data(days_back=days)

# 🧪 Feature Engineering
df["latitude"] = df["sourceLocation"].str.extract(r'([NS]\d+)')[0]
df["longitude"] = df["sourceLocation"].str.extract(r'([EW]\d+)')[0]

def convert_coord(coord):
    if pd.isna(coord): return None
    return float(coord[1:]) * (-1 if coord[0] in ['S', 'W'] else 1)

df["latitude"] = df["latitude"].apply(convert_coord)
df["longitude"] = df["longitude"].apply(convert_coord)
df["duration_minutes"] = pd.to_numeric(df["duration_minutes"], errors='coerce')

features = ["activeRegionNum", "duration_minutes", "latitude", "longitude"]
df = df.dropna(subset=features + ["classType"])

# 🚨 Safety Check
if df.empty:
    st.error("❌ No valid flare records available. Try increasing the date range.")
    st.stop()

# 🔀 Prepare training data
X = df[features]
y = df["classType"]
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 🧠 Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=42)
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# 📊 Evaluation
st.markdown("### 📈 Model Evaluation")
y_pred = model.predict(X_test)
accuracy = (y_pred == y_test).mean()
st.write(f"✅ **Accuracy:** `{accuracy * 100:.2f}%`")

with st.expander("🧾 Show Confusion Matrix"):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", 
                xticklabels=le.classes_, yticklabels=le.classes_)
    st.pyplot(fig)

# 🔮 Predict Section
st.header("🔮 Predict Flare Class")

mode = st.radio("Choose Input Method:", ["Manual Input", "Select from Live NASA Events"])

if mode == "Manual Input":
    st.subheader("📝 Enter Details:")
    ar = st.number_input("Active Region Number", min_value=0, value=13000)
    dur = st.number_input("Duration (minutes)", min_value=0.0, value=20.0)
    lat = st.number_input("Latitude (positive=N, negative=S)", value=10.0)
    lon = st.number_input("Longitude (positive=E, negative=W)", value=20.0)

    if st.button("🚀 Predict Now"):
        pred = model.predict([[ar, dur, lat, lon]])[0]
        label = le.inverse_transform([pred])[0]
        st.success(f"☀️ Predicted Flare Class: `{label}`")

else:
    st.subheader("📡 Select a Flare Event")
    selected = st.selectbox("Choose peak time:", df["peakTime"])
    row = df[df["peakTime"] == selected]
    input_vals = row[features].values[0]

    st.write("🧾 Input Values:")
    st.json(dict(zip(features, input_vals)))

    if st.button("🚀 Predict Selected Event"):
        pred = model.predict([input_vals])[0]
        label = le.inverse_transform([pred])[0]
        st.success(f"☀️ Predicted Flare Class: `{label}`")

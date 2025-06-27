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

st.set_page_config(page_title="SolarFlareSense AI", page_icon="☀️")
st.title("☀️ SolarFlareSense AI — NASA Solar Flare Intensity Predictor")
st.markdown("Predict whether a solar flare will be C, M, or X class using NASA's real-time DONKI data.")

# 🔭 Step 1: Load data
df = get_flare_data()
st.write(f"✅ Loaded {len(df)} flare records from NASA DONKI")

# 🧪 Step 2: Feature engineering
df["latitude"] = df["sourceLocation"].str.extract(r'([NS]\d+)')[0]
df["longitude"] = df["sourceLocation"].str.extract(r'([EW]\d+)')[0]

def convert_coord(coord):
    if pd.isna(coord): return None
    return float(coord[1:]) * (-1 if coord[0] in ['S', 'W'] else 1)

df["latitude"] = df["latitude"].apply(convert_coord)
df["longitude"] = df["longitude"].apply(convert_coord)

# 🔍 Step 3: Filter and prepare
features = ["activeRegionNum", "duration_minutes", "latitude", "longitude"]
df = df.dropna(subset=features + ["classType"])

X = df[features]
y = df["classType"]

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = (y_pred == y_test).mean()
st.markdown(f"📊 **Model Accuracy:** `{acc*100:.2f}%`")

# Confusion matrix
with st.expander("🧾 Show Confusion Matrix"):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
                xticklabels=le.classes_,
                yticklabels=le.classes_)
    st.pyplot(fig)

# 🔘 Input method
st.header("🔮 Predict Flare Class")

input_mode = st.radio("Select Input Type:", ["Manual Input", "Choose From Live Data"])

if input_mode == "Manual Input":
    st.subheader("🧪 Enter Flare Details")
    ar = st.number_input("Active Region Number", min_value=0, value=12987)
    dur = st.number_input("Duration (minutes)", min_value=1.0, value=20.0)
    lat = st.number_input("Latitude (positive for N, negative for S)", value=15.0)
    lon = st.number_input("Longitude (positive for E, negative for W)", value=30.0)

    if st.button("🔍 Predict Flare Class"):
        pred = model.predict([[ar, dur, lat, lon]])[0]
        st.success(f"☀️ Predicted Flare Class: `{le.inverse_transform([pred])[0]}`")

else:
    st.subheader("📡 Select Flare from Live NASA Data")
    selected = st.selectbox("Choose flare event", df["peakTime"].astype(str))
    row = df[df["peakTime"].astype(str) == selected]
    input_vals = row[features].values[0]

    st.write("📥 Features:")
    st.write(dict(zip(features, input_vals)))

    if st.button("🔍 Predict Selected Flare"):
        pred = model.predict([input_vals])[0]
        st.success(f"☀️ Predicted Flare Class: `{le.inverse_transform([pred])[0]}`")

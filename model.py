# model.py

import pandas as pd
from fetch_donki import get_flare_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# 1️⃣ Fetch live data from NASA
df = get_flare_data()
print("✅ Loaded solar flare data:", df.shape)

# 2️⃣ Feature engineering: extract lat/lon from sourceLocation
df["latitude"] = df["sourceLocation"].str.extract(r'([NS]\d+)')[0]
df["longitude"] = df["sourceLocation"].str.extract(r'([EW]\d+)')[0]

def convert_coord(coord):
    if pd.isna(coord): return None
    return float(coord[1:]) * (-1 if coord[0] in ['S', 'W'] else 1)

df["latitude"] = df["latitude"].apply(convert_coord)
df["longitude"] = df["longitude"].apply(convert_coord)

# 3️⃣ Final dataset
features = ["activeRegionNum", "duration_minutes", "latitude", "longitude"]
df = df.dropna(subset=features + ["classType"])

X = df[features]
y = df["classType"]

# 4️⃣ Encode labels (C, M, X → 0, 1, 2)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 5️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=42)

# 6️⃣ Train model
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# 7️⃣ Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"✅ Accuracy: {acc:.2f}")
print("\n🧾 Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# 8️⃣ Ready to use model and label encoder directly
print("✅ Model ready to use (in memory, not saved to .pkl)")

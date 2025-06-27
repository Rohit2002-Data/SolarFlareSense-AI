# fetch_donki.py

import requests
import pandas as pd
from datetime import datetime

def get_flare_data(start_date="2023-01-01", end_date="2023-12-31", api_key="DEMO_KEY"):
    """
    Fetch solar flare data from NASA DONKI API.
    Returns a cleaned pandas DataFrame with relevant features.
    """
    url = f"https://kauai.ccmc.gsfc.nasa.gov/DONKI/FLR?startDate={start_date}&endDate={end_date}&api_key={api_key}"
    
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"NASA API error: {response.status_code}")

    raw_data = response.json()
    
    # Extract relevant info
    records = []
    for flare in raw_data:
        if "classType" not in flare or flare["classType"] not in ["C", "M", "X"]:
            continue  # skip low/no-class events

        records.append({
            "beginTime": flare.get("beginTime"),
            "peakTime": flare.get("peakTime"),
            "endTime": flare.get("endTime"),
            "sourceLocation": flare.get("sourceLocation"),
            "activeRegionNum": flare.get("activeRegionNum"),
            "classType": flare.get("classType"),
        })

    df = pd.DataFrame(records)

    # Feature engineering
    df["beginTime"] = pd.to_datetime(df["beginTime"])
    df["peakTime"] = pd.to_datetime(df["peakTime"])
    df["endTime"] = pd.to_datetime(df["endTime"])
    df["duration_minutes"] = (df["endTime"] - df["beginTime"]).dt.total_seconds() / 60

    # Clean features
    df = df.dropna(subset=["activeRegionNum", "duration_minutes", "sourceLocation", "classType"])

    return df

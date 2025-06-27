# fetch_donki.py

import requests
import pandas as pd
from datetime import date, timedelta

def get_flare_data(days_back=365, api_key="DEMO_KEY"):
    end_date = date.today()
    start_date = end_date - timedelta(days=days_back)

    url = f"https://api.nasa.gov/DONKI/FLR?startDate={start_date}&endDate={end_date}&api_key={api_key}"
    print("🌐 Fetching:", url)

    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"NASA API error: {response.status_code}")

    raw_data = response.json()

    data = []
    for event in raw_data:
        data.append({
            "peakTime": event.get("peakTime"),
            "classType": event.get("classType"),
            "sourceLocation": event.get("sourceLocation"),
            "activeRegionNum": event.get("activeRegionNum"),
            "duration_minutes": event.get("duration")
        })

    return pd.DataFrame(data)

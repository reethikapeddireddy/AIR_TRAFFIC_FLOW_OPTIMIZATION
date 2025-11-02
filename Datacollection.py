# -------------------------------
# Data Collection
# -------------------------------

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate 1200 flight records
n = 1200
np.random.seed(42)

# Define possible values
origins = ['DEL', 'BOM', 'MAA', 'HYD', 'CCU', 'BLR']
dests   = ['DXB', 'LHR', 'CDG', 'SIN', 'JFK', 'SYD']
carriers = ['AI', '6E', 'UK', 'SG', 'G8']
runways = ['RW1', 'RW2', 'RW3', 'RW4']
traffic_levels = ['Low', 'Medium', 'High']
weather_conditions = ['Clear', 'Rain', 'Fog', 'Storm', 'Haze', 'Cloudy']

# Create random flight data
data = {
    "FlightID": [f"FL{1000+i}" for i in range(n)],
    "Date": [datetime(2025,1,1) + timedelta(days=np.random.randint(0,120)) for _ in range(n)],
    "Origin": np.random.choice(origins, n),
    "Dest": np.random.choice(dests, n),
    "DepDelay": np.random.randint(-5, 121, n),
    "ArrDelay": np.random.randint(-1, 241, n),
    "WeatherDelay": np.random.randint(0, 121, n),
    "Carrier": np.random.choice(carriers, n),
    "Runway": np.random.choice(runways, n),
    "TrafficLevel": np.random.choice(traffic_levels, n),
    "WeatherCondition": np.random.choice(weather_conditions, n)
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV
csv_path = "air_traffic_flow_dataset.csv"
df.to_csv(csv_path, index=False)

print("✅ Data collection completed successfully!")
print("📁 Dataset saved as:", csv_path)
print(df.head())
print("\nNumber of records:", len(df))

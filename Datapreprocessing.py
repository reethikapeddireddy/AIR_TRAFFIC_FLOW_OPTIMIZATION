# -------------------------------
# Step 2: Data Preprocessing
# -------------------------------


import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("air_traffic_flow_dataset.csv", parse_dates=["Date"])
print("✅ Dataset loaded successfully!")
print(df.info(), "\n")

#  Basic data check
print("Missing values per column:\n", df.isnull().sum(), "\n")
print("Unique values per column:\n", df.nunique(), "\n")

# Data Cleaning
df.columns = [c.strip().lower() for c in df.columns]   # lowercase column names
df = df.drop_duplicates()                              # remove duplicate rows
df['depdelay'] = df['depdelay'].clip(-10, 180)         # cap extreme delays
df['arrdelay'] = df['arrdelay'].clip(-10, 300)
df['weatherdelay'] = df['weatherdelay'].clip(0, 120)

#  Feature Engineering
df['weekday'] = df['date'].dt.weekday
df['month'] = df['date'].dt.month
df['is_delayed'] = (df['arrdelay'] > 15).astype(int)

#  Encode categorical columns
label_cols = ['origin', 'dest', 'carrier', 'runway', 'trafficlevel', 'weathercondition']
encoder = LabelEncoder()

for col in label_cols:
    df[col] = encoder.fit_transform(df[col])

print("✅ Label encoding completed!")

# Save preprocessed dataset
df.to_csv("air_traffic_flow_preprocessed.csv", index=False)
print("📁 Preprocessed dataset saved as: air_traffic_flow_preprocessed.csv")

#  Quick summary
print("\nSample data after preprocessing:")
print(df.head())
print("\nDataset shape:", df.shape)

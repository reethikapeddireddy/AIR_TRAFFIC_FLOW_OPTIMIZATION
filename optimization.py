# -------------------------------
# Step 4: Flight Delay Optimization
# -------------------------------

import pandas as pd
import pulp
import joblib

# Load model and data
model = joblib.load("rf_delay_prediction_model.joblib")
df = pd.read_csv("air_traffic_flow_preprocessed.csv")

print("✅ Model and data loaded!")

# Select sample for optimization (you can use full dataset)
sample_df = df.sample(100, random_state=42).reset_index(drop=True)

# Features used for prediction
features = [col for col in sample_df.columns if col not in ['flightid', 'date', 'arrdelay']]

# Predict delays using trained model
sample_df['PredictedDelay'] = model.predict(sample_df[features])

print("📊 Predicted delays added to data!")

# -------------------------------
# Define Optimization Problem
# -------------------------------
prob = pulp.LpProblem("Air_Traffic_Flow_Optimization", pulp.LpMinimize)

# Decision variables: 1 if flight is scheduled to take off, 0 otherwise
x = pulp.LpVariable.dicts("Flight", range(len(sample_df)), 0, 1, pulp.LpBinary)

# Objective: minimize total predicted delay
prob += pulp.lpSum([x[i] * sample_df.loc[i, 'PredictedDelay'] for i in range(len(sample_df))]), "TotalDelay"

# -------------------------------
# Constraints
# -------------------------------

# 1️⃣ Limit number of active flights (simulate runway capacity)
max_flights = 60
prob += pulp.lpSum([x[i] for i in range(len(sample_df))]) <= max_flights, "RunwayCapacity"

# 2️⃣ Ensure fair distribution among traffic levels
for level in sample_df['trafficlevel'].unique():
    indices = sample_df[sample_df['trafficlevel'] == level].index
    prob += pulp.lpSum([x[i] for i in indices]) >= 5, f"MinFlights_{level}"

print("✅ Constraints added!")

# -------------------------------
# Solve Optimization
# -------------------------------
prob.solve(pulp.PULP_CBC_CMD(msg=0))

# -------------------------------
# Display Results
# -------------------------------
sample_df['Selected'] = [int(x[i].value()) for i in range(len(sample_df))]
selected_flights = sample_df[sample_df['Selected'] == 1]

print("\n✅ Optimization complete!")
print(f"✈️ Selected flights for takeoff: {len(selected_flights)} out of {len(sample_df)}")
print(f"🚀 Total Predicted Delay (optimized): {selected_flights['PredictedDelay'].sum():.2f} mins")

# Save optimized results
selected_flights.to_csv("optimized_flights.csv", index=False)
print("💾 Optimized flight schedule saved as optimized_flights.csv")

# Preview top results
print("\n🔍 Sample optimized flights:")
print(selected_flights[['flightid', 'origin', 'dest', 'carrier', 'PredictedDelay']].head(10))

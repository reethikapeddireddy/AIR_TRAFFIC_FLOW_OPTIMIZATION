# -------------------------------
# Step 3: Model Training and Evaluation
# -------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Load preprocessed dataset
df = pd.read_csv("air_traffic_flow_preprocessed.csv")

print("✅ Preprocessed dataset loaded successfully!")
print("Shape:", df.shape)

# -------------------------------
# Train-Test Split
# -------------------------------
target = "arrdelay"

# Features = all columns except target and identifiers
features = [col for col in df.columns if col not in ["flightid", "date", target]]

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("📊 Training samples:", X_train.shape[0])
print("📊 Testing samples:", X_test.shape[0])

# -------------------------------
# Model Training
# -------------------------------
rf = RandomForestRegressor(
    n_estimators=200, random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)

print("✅ Model training completed!")

# -------------------------------
# Model Evaluation
# -------------------------------
preds = rf.predict(X_test)

mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print(f"\n📈 Model Performance:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# -------------------------------
# Save the trained model
# -------------------------------
joblib.dump(rf, "rf_delay_prediction_model.joblib")
print("💾 Model saved as rf_delay_prediction_model.joblib")

# Optional: View a few predictions
results = pd.DataFrame({
    "Actual": y_test.values[:10],
    "Predicted": preds[:10]
})
print("\n🔍 Sample Predictions:")
print(results)

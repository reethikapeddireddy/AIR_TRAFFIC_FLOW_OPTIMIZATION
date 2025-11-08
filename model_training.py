#--------------------------
# step6: model_training.py
#--------------------------
"""
Model Training for Air Traffic Flow Optimization
Trains Linear Regression, Random Forest, and Gradient Boosting regressors.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import math

# -------------------------
# Paths
# -------------------------
INPUT_CSV = "air_traffic_flow_preprocessed.csv"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# -------------------------
# Load dataset
# -------------------------
if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"Preprocessed file not found: {INPUT_CSV}")

df = pd.read_csv(INPUT_CSV)
print("✅ Loaded preprocessed data:", df.shape)

TARGET = "arrdelay"
if TARGET not in df.columns:
    raise KeyError(f"Target '{TARGET}' not found in dataset columns: {list(df.columns)}")

# Select numeric columns for features
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if TARGET in numeric_cols:
    numeric_cols.remove(TARGET)

X = df[numeric_cols]
y = df[TARGET]

print("📊 Using numeric features:", numeric_cols[:10], "...")
print("Total features:", len(numeric_cols))

# Fill NaNs
X = X.fillna(X.median())

# -------------------------
# Train-test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✅ Train shape: {X_train.shape} | Test shape: {X_test.shape}")

# -------------------------
# Define models
# -------------------------
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=200, random_state=42)
}

results = []

# -------------------------
# Train & Evaluate
# -------------------------
for name, model in models.items():
    print(f"\n🚀 Training {name} ...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"✅ {name}: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.3f}")

    joblib.dump(model, os.path.join(MODELS_DIR, f"{name}.joblib"))
    results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})

# -------------------------
# Save results
# -------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(MODELS_DIR, "model_results.csv"), index=False)
print("\n📁 Results saved to models/model_results.csv")
print(results_df)

print("\n🎯 Model training completed successfully!")

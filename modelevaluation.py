#-----------------------------
#Step 7: Model Evaluation & Hyperparameter Tuning
#----------------------------

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# -------------------------
# Paths
# -------------------------
PREPROCESSED_FILE = "air_traffic_flow_preprocessed.csv"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# -------------------------
# Load preprocessed data
# -------------------------
if not os.path.exists(PREPROCESSED_FILE):
    raise FileNotFoundError("Preprocessed dataset not found!")

df = pd.read_csv(PREPROCESSED_FILE)

TARGET = "arrdelay"
X = df.select_dtypes(include=[np.number]).drop(columns=[TARGET])
y = df[TARGET]

# Fill NaN values
X = X.fillna(X.median())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("✅ Dataset loaded for evaluation!")
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# -------------------------
# Load Trained Models
# -------------------------
model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".joblib")]
if not model_files:
    raise FileNotFoundError("No trained models found in 'models/' directory!")

print("\n📁 Found trained models:", model_files)

results = []

for file in model_files:
    name = file.replace(".joblib", "")
    model = joblib.load(os.path.join(MODELS_DIR, file))

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_test, preds)

    results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})

# -------------------------
# Compare Performance
# -------------------------
results_df = pd.DataFrame(results)
print("\n📊 Model Comparison Results:")
print(results_df)

# Identify best model
best_model_name = results_df.loc[results_df['R2'].idxmax(), 'Model']
print(f"\n🏆 Best model based on R²: {best_model_name}")

# -------------------------
# Hyperparameter Tuning for Best Model
# -------------------------
print("\n⚙️ Starting hyperparameter tuning...")

if best_model_name == "RandomForest":
    base_model = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 20],
        'min_samples_split': [2, 5, 10]
    }

elif best_model_name == "GradientBoosting":
    base_model = GradientBoostingRegressor(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 5]
    }

else:
    print("⚠️ Hyperparameter tuning is not required for Linear Regression.")
    base_model = None

# Run GridSearchCV
if base_model:
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=3,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    print(f"\n✅ Best Parameters for {best_model_name}: {grid_search.best_params_}")
    tuned_model = grid_search.best_estimator_

    # Evaluate tuned model
    preds_tuned = tuned_model.predict(X_test)
    mae_tuned = mean_absolute_error(y_test, preds_tuned)
    mse_tuned = mean_squared_error(y_test, preds_tuned)
    rmse_tuned = math.sqrt(mse_tuned)
    r2_tuned = r2_score(y_test, preds_tuned)

    print(f"\n🎯 Tuned {best_model_name} Results:")
    print(f"MAE = {mae_tuned:.2f}, RMSE = {rmse_tuned:.2f}, R² = {r2_tuned:.3f}")

    # Save optimized model
    tuned_path = os.path.join(MODELS_DIR, f"{best_model_name}_Tuned.joblib")
    joblib.dump(tuned_model, tuned_path)
    print(f"\n💾 Tuned model saved at: {tuned_path}")

print("\n✅ Step 7: Model Evaluation & Hyperparameter Tuning completed successfully!")

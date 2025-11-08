# ---------------------------------------------
# Step 5: Data Visualization (Optimized Version)
# ---------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load preprocessed dataset
df_pre = pd.read_csv("air_traffic_flow_preprocessed.csv")

# Optional: sample smaller subset for faster plotting
if len(df_pre) > 500:
    df_pre = df_pre.sample(500, random_state=42)

# Create folder for saving plots
os.makedirs("visualizations", exist_ok=True)

print("✅ Dataset loaded for visualization!")
print("📊 Number of rows:", len(df_pre), "\n")

# ---------------------------
# 1️⃣ Arrival Delay Distribution
# ---------------------------
plt.figure(figsize=(8, 5))
sns.histplot(df_pre['arrdelay'], bins=30, kde=True, color='steelblue')
plt.title("Distribution of Arrival Delay")
plt.xlabel("Arrival Delay (minutes)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("visualizations/delay_distribution.png")
plt.close()

# ---------------------------
# 2️⃣ Average Delay by Carrier
# ---------------------------
plt.figure(figsize=(8, 5))
sns.barplot(x='carrier', y='arrdelay', data=df_pre, palette='coolwarm', errorbar=None)
plt.title("Average Arrival Delay by Carrier")
plt.xlabel("Carrier")
plt.ylabel("Average Arrival Delay (min)")
plt.tight_layout()
plt.savefig("visualizations/avg_delay_carrier.png")
plt.close()

# ---------------------------
# 3️⃣ Traffic Level vs Arrival Delay
# ---------------------------
plt.figure(figsize=(8, 5))
sns.boxplot(x='trafficlevel', y='arrdelay', data=df_pre, hue='trafficlevel',
            legend=False, palette='viridis')
plt.title("Traffic Level vs Arrival Delay")
plt.xlabel("Traffic Level")
plt.ylabel("Arrival Delay (min)")
plt.tight_layout()
plt.savefig("visualizations/traffic_vs_delay.png")
plt.close()

# ---------------------------
# 4️⃣ Delay Trend by Month
# ---------------------------
plt.figure(figsize=(8, 5))
sns.lineplot(x='month', y='arrdelay', data=df_pre, marker='o', color='orange')
plt.title("Monthly Trend of Arrival Delay")
plt.xlabel("Month")
plt.ylabel("Average Delay (min)")
plt.tight_layout()
plt.savefig("visualizations/monthly_trend.png")
plt.close()

# ---------------------------
# 5️⃣ Top 10 Routes by Average Delay
# ---------------------------
# Recreate readable route column
df_pre['route'] = df_pre['origin'].astype(str) + "-" + df_pre['dest'].astype(str)

top_routes = (
    df_pre.groupby('route')['arrdelay']
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_routes.values, y=top_routes.index, palette='magma')
plt.title("Top 10 Routes by Average Arrival Delay")
plt.xlabel("Average Delay (min)")
plt.ylabel("Route")
plt.tight_layout()
plt.savefig("visualizations/top_routes.png")
plt.close()

# ---------------------------
#  ✅ Done
# ---------------------------
print("🎉 All visualizations created successfully!")
print("📁 Check the 'visualizations' folder for saved images.")

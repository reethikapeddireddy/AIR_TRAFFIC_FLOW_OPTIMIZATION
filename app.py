from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__, template_folder="templates")

# Load dataset (for dropdown values)
df = pd.read_csv("flight_data_large.csv")

# Unique dropdown values
origins = sorted(df["Origin"].dropna().unique().tolist())
destinations = sorted(df["Dest"].dropna().unique().tolist())
carriers = sorted(df["Carrier"].dropna().unique().tolist())
runways = sorted(df["Runway"].dropna().unique().tolist())
traffic_levels = sorted(df["TrafficLevel"].dropna().unique().tolist())
weather_conditions = sorted(df["WeatherCondition"].dropna().unique().tolist())


# -----------------------------
# HOME PAGE
# -----------------------------
@app.route("/")
def home():
    return render_template("home.html")


# -----------------------------
# PREDICT FORM PAGE
# -----------------------------
@app.route("/predict_form")
def predict_form():

    fields = [
        {"name": "Origin", "type": "select", "options": origins},
        {"name": "Dest", "type": "select", "options": destinations},
        {"name": "Carrier", "type": "select", "options": carriers},
        {"name": "Runway", "type": "select", "options": runways},
        {"name": "TrafficLevel", "type": "select", "options": traffic_levels},
        {"name": "WeatherCondition", "type": "select", "options": weather_conditions},
        {"name": "DepDelay", "type": "number", "placeholder": "Departure Delay"}
    ]

    return render_template("index.html", fields=fields)


# -----------------------------
# PREDICT RESULT
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():

    try:
        origin = request.form.get("Origin")
        dest = request.form.get("Dest")
        carrier = request.form.get("Carrier")
        runway = request.form.get("Runway")
        traffic = request.form.get("TrafficLevel")
        weather = request.form.get("WeatherCondition")
        depdelay = float(request.form.get("DepDelay"))

        # Simple delay prediction formula
        predicted = depdelay + 5

        result_text = f"Predicted Delay: {predicted:.2f} minutes"

    except:
        result_text = "Invalid Input"

    return render_template("index.html", fields=[], prediction_text=result_text)



# -----------------------------
# OPTIMIZATION FORM PAGE
# -----------------------------
@app.route("/optimization")
def optimization():

    fields = [
        {"name": "Origin", "type": "select", "options": origins},
        {"name": "Dest", "type": "select", "options": destinations},
        {"name": "TrafficLevel", "type": "select", "options": traffic_levels},
        {"name": "WeatherCondition", "type": "select", "options": weather_conditions}
    ]

    return render_template("optimization.html", fields=fields)



# -----------------------------
# OPTIMIZATION RESULT
# -----------------------------
@app.route("/optimize", methods=["POST"])
def optimize():

    origin = request.form.get("Origin")
    dest = request.form.get("Dest")
    traffic = request.form.get("TrafficLevel")
    weather = request.form.get("WeatherCondition")

    # Simple sample logic (you can replace with ML)
    result = f"Optimized Route from {origin} to {dest} with Traffic: {traffic} and Weather: {weather}"

    return render_template("optimization.html", fields=[], result_text=result)



# -----------------------------
# RUN APP
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)

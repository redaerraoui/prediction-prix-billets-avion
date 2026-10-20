"""Application Flask : prédiction du prix d'un billet d'avion.

Lancer d'abord `python analyse.py` pour créer flight_price_model.pkl, puis `python app.py`.
"""
import pickle
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request

FEATURES = ["airline_Air_India", "airline_GO_FIRST", "airline_Indigo", "airline_SpiceJet",
            "airline_Vistara", "arrival_time", "class_Economy", "days_left",
            "destination_city_Delhi", "destination_city_Mumbai", "duration",
            "source_city_Delhi", "source_city_Mumbai", "stops"]
AIRLINES = ["AirAsia", "Air_India", "GO_FIRST", "Indigo", "SpiceJet", "Vistara"]
CITIES = ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad", "Chennai"]
ARRIVAL = ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"]

MODEL_PATH = Path(__file__).with_name("flight_price_model.pkl")
model = pickle.load(open(MODEL_PATH, "rb")) if MODEL_PATH.exists() else None

app = Flask(__name__)


def encode(form):
    """Construit la ligne de variables dans l'ordre exact utilisé à l'entraînement.

    Une seule compagnie et une seule ville par champ : la compagnie AirAsia et les villes autres
    que Delhi et Mumbai correspondent aux modalités de référence (toutes les indicatrices à 0).
    """
    airline, src, dst = form["airline"], form["source_city"], form["destination_city"]
    row = {f"airline_{a}": int(airline == a) for a in AIRLINES[1:]}
    row.update({
        "arrival_time": ARRIVAL.index(form["arrival_time"]),
        "class_Economy": int(form["class"] == "Economy"),
        "days_left": int(form["days_left"]),
        "destination_city_Delhi": int(dst == "Delhi"),
        "destination_city_Mumbai": int(dst == "Mumbai"),
        "duration": float(form["duration"]),
        "source_city_Delhi": int(src == "Delhi"),
        "source_city_Mumbai": int(src == "Mumbai"),
        "stops": int(form["stops"]),
    })
    return pd.DataFrame([[row[f] for f in FEATURES]], columns=FEATURES, dtype=float)


@app.route("/", methods=["GET", "POST"])
def home():
    ctx = dict(airlines=AIRLINES, cities=CITIES, arrival=ARRIVAL, form=request.form, result=None, error=None)
    if request.method == "POST":
        if model is None:
            ctx["error"] = "Modèle introuvable : lancez d'abord python analyse.py."
        elif request.form["source_city"] == request.form["destination_city"]:
            ctx["error"] = "Choisissez deux villes différentes."
        elif request.form["class"] == "Business" and request.form["airline"] not in ("Air_India", "Vistara"):
            ctx["error"] = "Dans les données, seules Air India et Vistara proposent la classe affaires."
        else:
            try:
                ctx["result"] = round(float(model.predict(encode(request.form))[0]))
            except (KeyError, ValueError):
                ctx["error"] = "Valeurs invalides."
    return render_template("index.html", **ctx)


if __name__ == "__main__":
    app.run(debug=True)

"""Prédiction du prix des billets d'avion — reproduction du notebook du projet.

Étapes : encodage, sélection des variables (corrélation, SelectKBest, importance Random Forest),
comparaison de modèles de régression, puis entraînement du modèle léger utilisé par app.py.

    python analyse.py            # comparaison des modèles + sauvegarde de flight_price_model.pkl
"""
import gc
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              ExtraTreesRegressor, HistGradientBoostingRegressor)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Chargement
df = pd.read_csv("data/Clean_Dataset.csv")

# 2. Encodage ordinal des créneaux horaires et des escales
df["departure_time"] = df["departure_time"].map(
    {"Evening": 0, "Early_Morning": 1, "Morning": 2, "Afternoon": 3, "Night": 4, "Late_Night": 5})
df["arrival_time"] = df["arrival_time"].map(
    {"Early_Morning": 0, "Morning": 1, "Afternoon": 2, "Evening": 3, "Night": 4, "Late_Night": 5})
df["stops"] = df["stops"].map({"zero": 0, "one": 1, "two_or_more": 2})

# 3. Variables indicatrices (modalité de référence supprimée).
#    Le notebook encode aussi les 1 561 numéros de vol pour la sélection des variables ; ils sont
#    ensuite écartés du modèle final, on ne les crée donc pas ici (économie de mémoire).
enc = pd.get_dummies(df.drop(columns=["flight", "Unnamed: 0"]),
                     columns=["airline", "source_city", "destination_city", "class"], drop_first=True)

# 4. Variables retenues par l'union des trois méthodes de sélection (hors numéros de vol)
FEATURES = ["airline_Air_India", "airline_GO_FIRST", "airline_Indigo", "airline_SpiceJet",
            "airline_Vistara", "arrival_time", "class_Economy", "days_left",
            "destination_city_Delhi", "destination_city_Mumbai", "duration",
            "source_city_Delhi", "source_city_Mumbai", "stops"]

X = enc[FEATURES].astype(float)
y = enc["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Comparaison des modèles (le SVR, très lent sur 240 000 lignes, est omis ici)
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "KNN Regressor": KNeighborsRegressor(n_neighbors=5),
    "Extra Trees": ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "HistGradientBoosting (léger)": HistGradientBoostingRegressor(
        max_iter=500, max_leaf_nodes=31, learning_rate=0.1, random_state=42),
}

rows = []
for name in list(models):
    model = models[name]
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rows.append([name, mean_absolute_error(y_test, pred),
                 np.sqrt(mean_squared_error(y_test, pred)), r2_score(y_test, pred)])
    print(f"{name:30s} MAE = {rows[-1][1]:8.1f}  RMSE = {rows[-1][2]:8.1f}  R² = {rows[-1][3]:.4f}")
    if name != "HistGradientBoosting (léger)":
        models[name] = model = None  # libère la mémoire : les forêts complètes occupent plusieurs Go
        gc.collect()

print(pd.DataFrame(rows, columns=["Modèle", "MAE", "RMSE", "R2"]).sort_values("R2", ascending=False))

# 6. Sauvegarde du modèle léger (quelques Mo, contre environ 2 Go pour la Random Forest complète)
with open("flight_price_model.pkl", "wb") as f:
    pickle.dump(models["HistGradientBoosting (léger)"], f)
print("Modèle enregistré dans flight_price_model.pkl")

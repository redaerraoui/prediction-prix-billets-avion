# Prédiction du prix des billets d'avion

Projet de Big Data réalisé par **Reda Erraoui**, élève ingénieur à l'ENSA Agadir, filière Finance et Ingénierie Décisionnelle.

**[Essayer le simulateur en ligne](https://redaerraoui.github.io/simulateurs/prix-billets.html)** · **[Lire le notebook (PDF)](rapport.pdf)**

## Données

300 153 billets de vols intérieurs indiens réservés entre février et mars 2022 sur la plateforme EaseMyTrip. Ce jeu de données est public et disponible sur Kaggle sous le nom « Flight Price Prediction ».

- `data/Clean_Dataset.csv` : données nettoyées utilisées pour la modélisation. On y trouve 6 compagnies, 6 villes, la classe, le nombre d'escales, la durée, le nombre de jours avant le départ et le prix en roupies.
- `data/economy.csv` et `data/business.csv` : données brutes d'origine.

## Démarche

1. **Encodage.** Les créneaux horaires et le nombre d'escales sont encodés en ordinal. La compagnie, les villes et la classe sont transformées en variables indicatrices.
2. **Sélection des variables.** On prend l'union de trois méthodes : la corrélation avec le prix (|r| ≥ 0,05), SelectKBest (test F) et l'importance des variables d'une Random Forest. Les numéros de vol sont ensuite écartés. Il reste 14 variables.
3. **Comparaison de sept modèles de régression**, avec un découpage 80/20 (`random_state = 42`).

## Résultats (jeu de test, 60 031 billets)

| Modèle | MAE (₹) | RMSE (₹) | R² |
|---|---:|---:|---:|
| **Random Forest (200 arbres)** | **1 702** | **3 754** | **0,9727** |
| Extra Trees (200 arbres) | 1 985 | 4 197 | 0,9658 |
| Arbre de décision | 1 888 | 4 518 | 0,9604 |
| Gradient Boosting | 3 051 | 5 178 | 0,9480 |
| Régression linéaire | 4 523 | 6 917 | 0,9072 |
| KNN (k = 5) | 4 344 | 7 009 | 0,9047 |
| SVR sans standardisation | 15 367 | 25 839 | −0,2952 |
| HistGradientBoosting, 500 arbres de 31 feuilles (ajouté) | 2 196 | 3 805 | 0,9719 |

La classe porte à elle seule 88 % de l'importance des variables de la Random Forest, devant la durée (7 %) et le nombre de jours avant le départ (2 %).

## Remarques et améliorations (octobre 2026)

- **Taille du modèle.** La Random Forest sans limite de profondeur compte environ 162 000 nœuds par arbre, soit environ 2 Go une fois sauvegardée. Le `HistGradientBoostingRegressor` ajouté obtient un R² presque identique (0,9719 contre 0,9727) pour un fichier de 1,8 Mo. C'est ce modèle qu'utilisent l'application et le simulateur en ligne.
- **SVR.** Son R² négatif s'explique semble-t-il par l'absence de standardisation : le prix et la durée sont sur des échelles très différentes. Avec un `StandardScaler` (C = 10 000, ε = 100), le SVR atteint un R² de 0,95 sur un sous-échantillon de 20 000 billets.
- **Sélection des variables.** Elle est calculée sur l'ensemble des données, test compris. Refaite sur l'échantillon d'entraînement seul, la corrélation et SelectKBest retiennent exactement les mêmes variables : cette fuite d'information n'a donc pas d'effet ici.
- **Villes.** Avec `drop_first = True`, le modèle final ne distingue que Delhi, Mumbai et « autre ville ». Bangalore, Kolkata, Hyderabad et Chennai reçoivent donc la même prédiction, à variables égales.
- **Application.** Dans la version d'origine, les compagnies et les villes étaient des cases à cocher : on pouvait en sélectionner plusieurs à la fois, ce que le modèle n'a jamais vu. Ce sont désormais des listes déroulantes. L'application signale aussi qu'aucune compagnie autre qu'Air India et Vistara ne propose la classe affaires dans les données. Le gabarit `templates/index.html`, qui manquait, a été ajouté.

## Lancement

```bash
pip install -r requirements.txt
python analyse.py      # compare les modèles et crée flight_price_model.pkl
python app.py          # puis ouvrir http://127.0.0.1:5000
```

`analyse.py` entraîne deux forêts complètes : il faut compter quelques minutes et environ 4 Go de mémoire.

## Contact

Reda Erraoui · [Portfolio](https://redaerraoui.github.io) · redaerraouie@gmail.com

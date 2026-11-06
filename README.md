# Prévision des Prix des Denrées Alimentaires au Bénin (PAM / HDX 2002–2026)

Application interactive de visualisation et de prévision des prix des denrées sur les marchés béninois, basée sur les modèles Machine Learning (Random Forest & LightGBM) et les baselines naïves.

---

## 🚀 Lancement de l'Application

### 1. Installation des dépendances
```bash
pip install -r requirements.txt
```

### 2. Démarrage de l'interface Streamlit
Depuis la racine du projet :
```bash
streamlit run app.py
```
ou depuis le sous-dossier `benin-food-price-forecast` :
```bash
cd benin-food-price-forecast
streamlit run app.py
```

L'application s'ouvrira automatiquement à l'adresse `http://localhost:8501`.

---

## 🖥️ Les Deux Vues Disponibles

### 1. 🔮 Prévision des prix des denrées au Bénin (Vue Inférence)
* **Barre latérale (Paramètres)** :
  * Sélection du produit (Maïs blanc, Riz importé, Haricot blanc, Gari, Tomate, etc.).
  * Sélection du marché (Dantokpa, Bohicon, Parakou, Natitingou, etc.).
  * Choix de l'horizon de prévision (1 mois, 2 mois, 3 mois).
  * Sélection du modèle (Modèle recommandé, Random Forest, LightGBM, Naïf).
  * Comparaison avec d'autres marchés sous forme de cases à cocher.
* **Indicateurs clés (KPIs)** :
  * Dernier prix observé (février 2026).
  * Prix prévu (mars 2026 / M+1).
  * Taux de variation prévu en %.
* **Graphique interactif** :
  * Ligne verte continue : Historique réel observé.
  * Ligne orange pointillée : Projection prévisionnelle.
  * Zone ombrée : Intervalle de confiance calculé via la MAE du modèle.
* **Tableaux d'analyse** :
  * Tableau gauche : Prévisions mois par mois (H1, H2, H3 avec bornes).
  * Tableau droit : Comparaison multi-marchés pour le même produit.

### 2. 📊 À propos des modèles (Vue Gouvernance & Métriques)
* **Indicateurs de déploiement** :
  * Modèle actuellement en service (`random_forest v1.0.0` ou `LightGBM`).
  * Date du dernier réentraînement.
  * Erreur globale sur le jeu de test (MAE et MAPE).
* **Tableau comparatif (MAE)** :
  * Évaluation par produit des modèles : Naïf, Naïf saisonnier, SARIMA, LightGBM et Random Forest.
  * Mise en avant en vert du modèle final retenu.

# Prévision des Prix des Denrées Alimentaires au Bénin (PAM / HDX 2002–2026)

Application interactive de visualisation et de prévision des prix des denrées sur les marchés béninois, basée sur les modèles Machine Learning (Random Forest & LightGBM), les baselines statistiques (SARIMA) et les modèles de référence naïfs.

---

## 🚀 Lancement Rapide de l'Application

### 1. Installation des dépendances
```bash
pip install -r requirements.txt
```

### 2. Démarrage de l'interface Streamlit
Depuis la racine du projet :
```bash
cd benin-food-price-forecast
streamlit run app.py
```

L'application s'ouvrira automatiquement à l'adresse `http://localhost:8501`.

---

## 🖥️ Les Pages de l'Interface

### 1. 🔮 Prévision des prix des denrées au Bénin (Vue Inférence)
* **Barre latérale (Paramètres)** :
  * Sélection du produit (filtrée strictement selon les denrées historiques validées : Maïs blanc, Riz importé, Haricot blanc, Gari, Tomate, etc.).
  * Sélection dynamique du marché d'intérêt (parmi ceux répertoriés pour le produit).
  * Choix de l'horizon de projection (1 mois, 2 mois, 3 mois).
  * Sélection du modèle (Recommandé / Random Forest / LightGBM / Baseline Naïve).
  * Comparaison multi-marchés dynamique avec sélection multiple.
* **Indicateurs clés (KPIs)** :
  * Dernier prix observé (historique réel).
  * Prix prévu à l'horizon choisi ($M+1, M+2, M+3$).
  * Taux de variation prévu en %.
* **Graphique interactif** :
  * Ligne continue : Historique réel observé.
  * Ligne pointillée : Projection prévisionnelle.
  * Zone ombrée : Intervalle de confiance calculé via la MAE du modèle.
* **Tableaux d'analyse** :
  * Prévisions détaillées mois par mois avec bornes d'incertitude.
  * Comparaison multi-marchés simultanée sur le même horizon temporel.

### 2. 📊 À propos des modèles (Vue Gouvernance & Métriques)
* **Indicateurs de déploiement** :
  * Modèle actuellement en service (`random_forest v1.0.0` ou `LightGBM`).
  * Date du dernier réentraînement et date max des données historiques.
  * Erreur globale sur le jeu de test holdout (MAE, RMSE et MAPE).
* **Tableau comparatif des performances (MAE)** :
  * Évaluation comparative par produit : Naïf dernier mois, Naïf saisonnier (M-12), SARIMA, LightGBM et Random Forest.
  * Mise en évidence visuelle du modèle champion retenu en production.

### 3. 🔎 Analyse Approfondie Multi-Marchés
* **Sélection du produit** pour une analyse transversale complète sur l'ensemble des marchés nationaux.
* **Moteur d'analyse textuelle & expressions régulières (Regex)** :
  * Extraction et qualification automatique des points de vente les plus compétitifs (*Marchés les plus abordables*).
  * Identification des marchés sous tension inflationniste (*Marchés les plus chers / à surveiller*).
  * Détection des marchés stables vs à forte volatilité saisonnière.
* **Synthèse tabulaire & métriques de dispersion** (moyenne, écart-type, médiane, minimum et maximum).

---

## 🔄 Réentraînement Automatique Mensuel (MLOps Workflow)

Le projet intègre un pipeline complet d'automatisation MLOps pour le réentraînement mensuel sans interruption de service, garantissant que les modèles en production restent continuellement calibrés sur les dernières données du Programme Alimentaire Mondial (PAM/HDX).

```
                      [Sources Externes]
              (HDX CSV / CKAN API / HuggingFace)
                              │
                              ▼
                   ┌──────────────────────┐
                   │ src/data_download.py │ ◄── Repli automatique sur cache local
                   └──────────┬───────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │ src/data_validation.py │ ──► Échec ? ──► Arrêt propre & Log
                  └───────────┬────────────┘
                              │ Données saines
                              ▼
                   ┌──────────────────────┐
                   │   src/features.py    │ (prepare_data & build_features)
                   └──────────┬───────────┘
                              │
                              ▼
            ┌───────────────────────────────────┐
            │  Holdout Split (N derniers mois)  │
            └─────────────────┬─────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │ Entraînement Modèle Candidat  │
              │  (Dossier models_candidate/)  │
              └───────────────┬───────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │     Évaluation Comparative    │
              │     Champion vs Challenger    │
              └───────────────┬───────────────┘
                              │
            ┌─────────────────┴─────────────────┐
            ▼                                   ▼
[Candidat MEILLEUR (ou --force)]      [Candidat NON MEILLEUR]
            │                                   │
   Archivage Production                         │
   dans models/archive/v{old}/                  │
            │                                   │
   Promotion du Candidat                        │
   dans models/ & bump version                  │
            │                                   │
            └─────────────────┬─────────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ models/retrain_log.jsonl│
                 │   models/metadata.json  │
                 └─────────────────────────┘
```

### 1. Architecture des Modules
* `config/retrain.yaml` : Configuration centralisée externalisée (URLs sources HDX, paramètres de validation, holdout de 6 mois, seuils de décision).
* `src/data_download.py` : Téléchargement résilient avec chaîne de repli en cascade (*HDX Direct $\to$ HDX CKAN API $\to$ Miroir HuggingFace $\to$ Cache local `data/raw/`*).
* `src/data_validation.py` : Validation stricte des données brutes et préparées (colonnes obligatoires, types, prix strictement positifs, filtre d'aberrations, non-régression de l'antériorité temporelle).
* `src/model_registry.py` : Gestionnaire de registre de modèles (versionnage sémantique `v1.0.0`, archivage automatique dans `models/archive/v{version}/`, promotion sans écrasement préalable, audit trail dans `models/retrain_log.jsonl`, fonction de rollback instantané).
* `scripts/retrain.py` : Orchestrateur CLI principal exécutant les étapes a à h du workflow Champion/Challenger.
* `scripts/replay_history.py` : Script de démonstration et backtesting séquentiel simulant le passage du temps mois par mois sur un dossier isolé (`models_simulation/`).

### 2. Règle de Décision Champion / Challenger
Pour être promu en production, le modèle candidat doit satisfaire la règle de décision stricte suivante :
1. **Performance globale** : $\text{MAE}_{\text{candidat}} \le \text{MAE}_{\text{champion}} \times (1 + \epsilon)$ avec $\epsilon = 0.0$ (ou seuil configuré dans `retrain.yaml`).
2. **Garde-fou par produit** : Aucune dégradation majeure sur un produit critique ($\text{MAE}_{\text{produit, candidat}} \le \text{MAE}_{\text{produit, champion}} \times 1.15$, tolérance max 15%).
3. **Sécurité opérationnelle** : Le modèle en production n'est jamais écrasé avant la validation complète. En cas de rejet, le champion actuel reste en place, le log enregistre le motif, et le code de sortie est 0.

### 3. Commandes Utiles (CLI)

#### Exécuter le réentraînement automatique
```bash
# Exécution standard avec source automatique (HDX / fallback)
python scripts/retrain.py

# Mode simulation sans toucher aux modèles de production
python scripts/retrain.py --dry-run

# Forcer le remplacement pour une démonstration
python scripts/retrain.py --force

# Forcer une source locale avec fenêtre de holdout spécifique (ex: 4 mois)
python scripts/retrain.py --source local --holdout-months 4
```

#### Rejeu historique séquentiel (Backtest / Démo)
```bash
# Simule 3 mois successifs d'arrivée de données et de réentraînement
python scripts/replay_history.py --start-date 2024-06-01 --steps 3 --output-dir models_simulation
```

#### Procédure de Rollback
Si un modèle déployé doit être rétrogradé en urgence vers une version précédente :
```python
from src.model_registry import rollback_to_version

# Restaure instantanément la version v1.0.0 depuis models/archive/v1.0.0/
rollback_to_version("1.0.0", models_dir="models", archive_dir="models/archive")
```

### 4. Automatisation CI/CD (GitHub Actions)
Le workflow est automatisé via `.github/workflows/retrain.yml` :
* **Planification** : Déclenchement automatique mensuel via cron (`0 4 1 * *`, chaque 1er du mois à 04:00 UTC).
* **Manuel** : Déclenchement à la demande (`workflow_dispatch`) avec paramètres configurables (`force`, `dry-run`, `holdout-months`).
* **Cache & Optimisation** : Mise en cache `pip` pour accélérer l'exécution.
* **Commit conditionnel** : Si et seulement si un nouveau modèle est promu et validé, les fichiers de métadonnées et modèles compressés sont automatiquement versionnés et commités avec le tag `[skip ci]`.
* **Respect des quotas GitHub** : Les modèles compressés joblib (`compress=3`) pèsent moins de 2 Mo au total, bien en dessous de la limite des 100 Mo de GitHub.

### 5. Tests Automatisés
La suite de tests unitaires et d'intégration garantit la robustesse du cycle de vie des modèles :
```bash
pytest tests/test_retrain.py -v
```
Couverture testée :
* `test_validation_accepts_valid_data` : validation de schémas valides.
* `test_validation_rejects_missing_columns` : détection d'anomalies de schéma.
* `test_validation_rejects_negative_prices` : intégrité des valeurs de prix.
* `test_validation_rejects_time_regression` : détection de régressions d'antériorité.
* `test_decision_rule_champion_challenger` : promotion conforme du champion.
* `test_decision_rule_product_degradation` : rejet automatique en cas de dégradation par produit.
* `test_bump_version_semantics` : incrémentation sémantique rigoureuse (`v1.0.0` $\to$ `v1.1.0`).
* `test_archive_and_rollback` : cycle complet d'archivage et de restauration sans perte.

---

## 📂 Structure du Répertoire

```
benin-food-price-forecast/
├── .github/
│   └── workflows/
│       └── retrain.yml          # GitHub Actions cron mensuel
├── config/
│   └── retrain.yaml             # Paramètres externalisés du réentraînement
├── data/
│   ├── raw/                     # Données brutes téléchargées et cache
│   └── prix_propres.parquet     # Dataset nettoyé et standardisé
├── models/
│   ├── metadata.json            # Version, date d'entraînement, métriques
│   ├── params.json              # Hyperparamètres fixés
│   ├── retrain_log.jsonl        # Journal d'audit append-only des réentraînements
│   ├── random_forest_h*.joblib  # Modèles Random Forest en production (h1, h2, h3)
│   ├── lightgbm_h*.joblib       # Modèles LightGBM en production (h1, h2, h3)
│   └── archive/                 # Versions historiques archivées (v1.0.0, ...)
├── scripts/
│   ├── retrain.py               # Script principal de réentraînement CLI
│   └── replay_history.py        # Rejeu historique séquentiel (backtest)
├── src/
│   ├── data_download.py         # Module de téléchargement multi-sources résilient
│   ├── data_validation.py       # Validation stricte des données brutes et préparées
│   ├── model_registry.py        # Gestion des versions, comparaison et rollback
│   ├── features.py              # Préparation des données et feature engineering
│   └── models.py                # Entraînement et inférence des modèles
├── tests/
│   ├── conftest.py              # Configuration pytest et injection du sys.path
│   └── test_retrain.py          # 8 tests unitaires et d'intégration
├── app.py                       # Interface utilisateur Streamlit (3 pages)
├── requirements.txt             # Dépendances Python du projet
└── README.md                    # Documentation complète du projet
```

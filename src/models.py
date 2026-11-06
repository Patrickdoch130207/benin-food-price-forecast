"""Module d'entraînement, sélection, évaluation et export des modèles Machine Learning.

Supporte Random Forest et LightGBM pour les horizons H1, H2 et H3.
Produit les artefacts .joblib attendus pour le déploiement et le fichier metadata.json.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor

from .metrics import compute_metrics, evaluate_baseline_naive

RANDOM_STATE = 42

# Vérification dynamique de LightGBM
try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


def get_default_feature_names(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Identifie automatiquement les variables catégorielles et numériques explicatives."""
    colonnes_exclues = {
        "date", "price", "target_1m", "target_2m", "target_3m",
        "prix_historique", "prix_hist"
    }
    feature_cols = [c for c in df.columns if c not in colonnes_exclues]
    feature_cols = [c for c in feature_cols if not c.lower().startswith("unnamed")]

    categorical_cols = [
        c for c in feature_cols
        if pd.api.types.is_string_dtype(df[c])
        or pd.api.types.is_object_dtype(df[c])
        or isinstance(df[c].dtype, pd.CategoricalDtype)
    ]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    return categorical_cols, numeric_cols


def build_pipeline(
    algo: str = "random_forest",
    params: Optional[Dict[str, Any]] = None,
    categorical_cols: Optional[List[str]] = None,
    numeric_cols: Optional[List[str]] = None
) -> Pipeline:
    """Construit un pipeline Scikit-Learn combinant prétraitement et modèle d'estimation.

    Parameters
    ----------
    algo : str
        'random_forest' ou 'lightgbm'.
    params : dict, optional
        Hyperparamètres du modèle.
    categorical_cols : list of str
        Colonnes catégorielles.
    numeric_cols : list of str
        Colonnes numériques.

    Returns
    -------
    Pipeline
        Pipeline complet autonome pour l'inférence.
    """
    params = params or {}

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
            ]), categorical_cols or []),
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median"))
            ]), numeric_cols or [])
        ],
        remainder="drop"
    )

    if algo == "random_forest":
        model_params = {
            "n_estimators": params.get("n_estimators", 150),
            "max_depth": params.get("max_depth", 15),
            "min_samples_leaf": params.get("min_samples_leaf", 2),
            "max_features": params.get("max_features", "sqrt"),
            "random_state": RANDOM_STATE,
            "n_jobs": 1
        }
        estimator = RandomForestRegressor(**model_params)
    elif algo == "lightgbm":
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM n'est pas installé. Exécuter 'pip install lightgbm'.")
        model_params = {
            "n_estimators": params.get("n_estimators", 300),
            "learning_rate": params.get("learning_rate", 0.05),
            "max_depth": params.get("max_depth", 6),
            "num_leaves": params.get("num_leaves", 31),
            "subsample": params.get("subsample", 0.85),
            "colsample_bytree": params.get("colsample_bytree", 0.85),
            "random_state": RANDOM_STATE,
            "n_jobs": 1,
            "verbosity": -1
        }
        estimator = LGBMRegressor(**model_params)
    else:
        raise ValueError(f"Algorithme inconnu : {algo}")

    return Pipeline([("preprocessing", preprocessor), ("model", estimator)])


def optimize_model_rolling_origin(
    df: pd.DataFrame,
    algo: str,
    horizon: int,
    configs: List[Dict[str, Any]],
    plis: List[Dict[str, Any]],
    categorical_cols: List[str],
    numeric_cols: List[str]
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Sélectionne la meilleure configuration sur les plis Rolling Origin selon la MAE."""
    target_col = f"target_{horizon}m"
    all_features = categorical_cols + numeric_cols

    search_rows = []
    best_config = None
    best_mae = float("inf")

    for idx, params in enumerate(configs, 1):
        fold_maes = []
        for p in plis:
            tr = df.loc[p["train_idx"]].dropna(subset=[target_col])
            va = df.loc[p["val_idx"]].dropna(subset=[target_col])

            if len(tr) == 0 or len(va) == 0:
                continue

            pipe = build_pipeline(algo, params, categorical_cols, numeric_cols)
            pipe.fit(tr[all_features], tr[target_col])
            pred = pipe.predict(va[all_features])
            fold_maes.append(compute_metrics(va[target_col], pred)["MAE"])

        mean_mae = float(np.mean(fold_maes)) if fold_maes else np.nan
        row = {"configuration": idx, "algo": algo, "horizon": f"H{horizon}", "MAE_moyenne": round(mean_mae, 2)}
        row.update(params)
        search_rows.append(row)

        if mean_mae < best_mae:
            best_mae = mean_mae
            best_config = params

    results_df = pd.DataFrame(search_rows).sort_values("MAE_moyenne")
    return best_config, results_df


def train_model_for_horizon(
    train_df: pd.DataFrame,
    algo: str,
    horizon: int,
    params: Optional[Dict[str, Any]] = None,
    categorical_cols: Optional[List[str]] = None,
    numeric_cols: Optional[List[str]] = None
) -> Tuple[Pipeline, List[str]]:
    """Entraîne un pipeline complet pour un horizon donné sur l'ensemble d'entraînement.

    Parameters
    ----------
    train_df : pd.DataFrame
        Table d'entraînement (< 2025).
    algo : str
        'random_forest' ou 'lightgbm'.
    horizon : int
        1, 2 ou 3.
    params : dict, optional
        Paramètres du modèle.

    Returns
    -------
    pipeline, features_list : Tuple[Pipeline, List[str]]
    """
    if categorical_cols is None or numeric_cols is None:
        cat_c, num_c = get_default_feature_names(train_df)
        categorical_cols = categorical_cols or cat_c
        numeric_cols = numeric_cols or num_c

    target_col = f"target_{horizon}m"
    train_clean = train_df.dropna(subset=[target_col]).copy()
    features = categorical_cols + numeric_cols

    pipe = build_pipeline(algo, params, categorical_cols, numeric_cols)
    pipe.fit(train_clean[features], train_clean[target_col])

    return pipe, features


def export_model_artifact(
    pipeline: Pipeline,
    features: List[str],
    algo: str,
    horizon: int,
    model_dir: str = "models"
) -> str:
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    filename = f"{algo}_h{horizon}.joblib"
    filepath = Path(model_dir) / filename

    artifact = {
        "pipeline": pipeline,
        "features": list(features),
        "algo": algo,
        "horizon": int(horizon)
    }

    joblib.dump(artifact, str(filepath))
    return str(filepath)


def generate_metadata(
    model_dir: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    selected_models: Dict[int, str],
    metrics_by_model: Dict[str, Dict[str, Any]],
    exported_files: List[str],
    baseline_metrics: Optional[Dict[str, Dict[str, Any]]] = None,
    version: str = "1.0.0"
) -> str:
    """Génère le fichier metadata.json traçant les métriques et fichiers exportés.

    Le format est aligné :
    - clé "modele_final" (pas "modele_final_global")
    - clé "metriques" structurée par algo > horizon
    """
    metadata_path = Path(model_dir) / "metadata.json"

    # Déterminer la famille globale majoritaire
    algos = list(selected_models.values())
    modele_global = max(set(algos), key=algos.count) if algos else "random_forest"

    unites = sorted(train_df["unit"].dropna().astype(str).unique().tolist()) if "unit" in train_df.columns else []

    # Restructurer les métriques : de "algo_hN" -> {algo: {hN: {...}}}
    metriques_structurees = {}
    for key, scores in metrics_by_model.items():
        # key est de la forme "random_forest_h1" ou "lightgbm_h2"
        parts = key.rsplit("_h", 1)
        if len(parts) == 2:
            algo_name, h_str = parts
            h_key = f"h{h_str}"
        else:
            algo_name = key
            h_key = "h1"
        if algo_name not in metriques_structurees:
            metriques_structurees[algo_name] = {}
        metriques_structurees[algo_name][h_key] = scores

    # Ajouter les baselines naïves
    if baseline_metrics:
        for key, scores in baseline_metrics.items():
            if key not in metriques_structurees:
                metriques_structurees[key] = {}
            metriques_structurees[key].update(scores)

    metadata = {
        "version": version,
        "date_entrainement": pd.Timestamp.today().strftime("%Y-%m-%d"),
        "donnees_jusqu_a": str(train_df["date"].max().date())[:7],
        "date_debut_train": str(train_df["date"].min().date()),
        "date_fin_train": str(train_df["date"].max().date()),
        "date_debut_test": str(test_df["date"].min().date()),
        "date_fin_test": str(test_df["date"].max().date()),
        "modele_final": modele_global,
        "choix_par_horizon": {f"h{h}": algo for h, algo in selected_models.items()},
        "horizons": [1, 2, 3],
        "metriques": metriques_structurees,
        "unites": unites,
        "fichiers_modeles_exportes": exported_files,
        "notes": [
            "Modèles entraînés sans fuite sur les données antérieures à 2025.",
            "Évaluation finale réalisée sur la période scellée 2025–2026.",
            "Pipeline complet autonome comprenant imputation et encodage ordinal pour l'API."
        ]
    }

    with open(str(metadata_path), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return str(metadata_path)

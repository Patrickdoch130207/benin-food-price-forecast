"""Module d'entraînement, sélection, évaluation et export des modèles Machine Learning.

Aligné sur l'EDA unifiée (01_cleaning.ipynb, 02_eda.ipynb) et le Feature Engineering
unifié (03_features.ipynb, src/features.py : 10 variables explicatives + commodity + market).
Supporte Random Forest et LightGBM pour les horizons H1, H2 et H3 avec :
- Ancrage métier sans fuite (prix_m1 -> prix_dernier_connu -> med_nat_m1 * ecart_marche)
- Cible en log-variation (log(prix / ancre)) pour traiter l'asymétrie (tomate, oignon, riz)
- Fonction de perte L1 (MAE / absolute_error) robuste aux variations extrêmes
- Entraînement à partir de 2020 (évite 2019 où prix_m12 est absent)
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import json
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor

from .features import FEATURES, NOMS_PRODUITS
from .metrics import (
    compute_metrics,
    evaluate_baseline_naive,
    evaluate_baseline_seasonal_naive,
    get_naive_predictions,
    get_seasonal_naive_predictions,
)

warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42

# Vérification dynamique de LightGBM
try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class AnchorResidualRegressor(BaseEstimator, RegressorMixin):
    """Estimateur Scikit-Learn prédisant la log-variation autour de l'ancre métier passée.

    Conforme aux enseignements de 02_eda.ipynb (sections 1.1, 2.4, 2.5, 4.3) et 03_features.ipynb (section 5.4) :
    1. Lorsque `prix_m1` est vide (15.5 % des cas), le repère passé sans fuite est `prix_dernier_connu`
       (disponible jusqu'à M-3), puis `med_nat_m1 * ecart_marche`, puis `med_nat_m1`.
    2. Le passage au logarithme `log(y / ancre)` stabilise la queue de distribution très étalée
       de la tomate (asymétrie 1.92 -> -0.24) et de l'oignon (2.87 -> -0.11).
    """

    def __init__(self, base_pipeline: Pipeline = None, shrinkage: float = 0.7):
        self.base_pipeline = base_pipeline
        self.shrinkage = shrinkage

    def _compute_anchor(self, X: pd.DataFrame) -> np.ndarray:
        est_nat = X["med_nat_m1"] * X["ecart_marche"].fillna(1.0)
        anchor = (
            X["prix_m1"]
            .fillna(X["prix_dernier_connu"])
            .fillna(est_nat)
            .fillna(X["med_nat_m1"])
        )
        fallback = getattr(self, "global_median_", 350.0)
        return np.clip(anchor.fillna(fallback).to_numpy(dtype=float), 1.0, None)

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]):
        y_arr = np.asarray(y, dtype=float)
        self.global_median_ = float(np.nanmedian(y_arr))
        anchor = self._compute_anchor(X)
        log_ratio = np.log(np.clip(y_arr, 1.0, None) / anchor)

        self.model_ = clone(self.base_pipeline)
        self.model_.fit(X, log_ratio)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        anchor = self._compute_anchor(X)
        pred_log_ratio = self.model_.predict(X) * self.shrinkage
        return anchor * np.exp(pred_log_ratio)


def get_default_feature_names(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Renvoie les colonnes catégorielles et les 10 features numériques unifiées de src/features.py."""
    categorical_cols = [c for c in ["commodity", "market"] if c in df.columns]
    numeric_cols = [c for c in FEATURES if c in df.columns]
    return categorical_cols, numeric_cols


def build_pipeline(
    algo: str = "random_forest",
    params: Optional[Dict[str, Any]] = None,
    categorical_cols: Optional[List[str]] = None,
    numeric_cols: Optional[List[str]] = None
) -> AnchorResidualRegressor:
    """Construit le pipeline complet combinant prétraitement, ancrage log-variation et arbre L1 (MAE)."""
    params = params or {}
    categorical_cols = categorical_cols if categorical_cols is not None else ["commodity", "market"]
    numeric_cols = numeric_cols if numeric_cols is not None else list(FEATURES)

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
            ]), categorical_cols),
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median"))
            ]), numeric_cols)
        ],
        remainder="drop"
    )

    if algo == "random_forest":
        shrinkage = float(params.get("shrinkage", 0.65))
        model_params = {
            "criterion": params.get("criterion", "absolute_error"),
            "n_estimators": int(params.get("n_estimators", 150)),
            "max_depth": int(params.get("max_depth", 7)),
            "min_samples_leaf": int(params.get("min_samples_leaf", 12)),
            "max_features": params.get("max_features", 0.7),
            "random_state": RANDOM_STATE,
            "n_jobs": -1
        }
        estimator = RandomForestRegressor(**model_params)
    elif algo == "lightgbm":
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM n'est pas installé. Exécuter 'pip install lightgbm'.")
        shrinkage = float(params.get("shrinkage", 0.85))
        model_params = {
            "objective": params.get("objective", "mae"),
            "n_estimators": int(params.get("n_estimators", 120)),
            "learning_rate": float(params.get("learning_rate", 0.03)),
            "max_depth": int(params.get("max_depth", 4)),
            "num_leaves": int(params.get("num_leaves", 12)),
            "min_child_samples": int(params.get("min_child_samples", 30)),
            "subsample": float(params.get("subsample", 0.8)),
            "colsample_bytree": float(params.get("colsample_bytree", 0.8)),
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbosity": -1
        }
        estimator = LGBMRegressor(**model_params)
    else:
        raise ValueError(f"Algorithme inconnu : {algo}")

    inner_pipe = Pipeline([("preprocessing", preprocessor), ("model", estimator)])
    return AnchorResidualRegressor(base_pipeline=inner_pipe, shrinkage=shrinkage)


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
    best_config = configs[0] if configs else {}
    best_mae = float("inf")

    for idx, params in enumerate(configs, 1):
        fold_maes = []
        for p in plis:
            tr = df.loc[df.index.intersection(p["train_idx"])].dropna(subset=[target_col])
            va = df.loc[df.index.intersection(p["val_idx"])].dropna(subset=[target_col])
            tr = tr[tr["date"] >= "2020-01-01"]

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

        if not np.isnan(mean_mae) and mean_mae < best_mae:
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
) -> Tuple[AnchorResidualRegressor, List[str]]:
    """Entraîne un pipeline complet pour un horizon donné sur l'ensemble d'entraînement (>= 2020)."""
    if categorical_cols is None or numeric_cols is None:
        cat_c, num_c = get_default_feature_names(train_df)
        categorical_cols = categorical_cols or cat_c
        numeric_cols = numeric_cols or num_c

    target_col = f"target_{horizon}m" if f"target_{horizon}m" in train_df.columns else "prix"
    train_clean = train_df.dropna(subset=[target_col]).copy()
    # Comme documenté en 5.4 de 03_features.ipynb : on démarre l'entraînement en 2020
    if "date" in train_clean.columns:
        train_2020 = train_clean[pd.to_datetime(train_clean["date"]) >= "2020-01-01"]
        if len(train_2020) >= 100:
            train_clean = train_2020

    features = categorical_cols + numeric_cols
    pipe = build_pipeline(algo, params, categorical_cols, numeric_cols)
    pipe.fit(train_clean[features], train_clean[target_col])

    return pipe, features


def export_model_artifact(
    pipeline: Any,
    features: List[str],
    algo: str,
    horizon: int,
    model_dir: str = "models"
) -> str:
    """Sauvegarde l'artefact au format standard {'pipeline', 'features', 'algo', 'horizon'}."""
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    filename = f"{algo}_h{horizon}.joblib"
    filepath = Path(model_dir) / filename

    artifact = {
        "pipeline": pipeline,
        "features": list(features),
        "algo": algo,
        "horizon": int(horizon)
    }

    joblib.dump(artifact, str(filepath), compress=3)
    return str(filepath)


def save_params_json(best_configs: Dict[Tuple[str, int], Dict[str, Any]], model_dir: str = "models") -> str:
    """Sauvegarde les hyperparamètres optimaux dans models/params.json."""
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    params_path = Path(model_dir) / "params.json"
    structured = {"random_forest": {}, "lightgbm": {}}
    for (algo, h), cfg in best_configs.items():
        if algo not in structured:
            structured[algo] = {}
        structured[algo][f"h{h}"] = cfg
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(structured, f, ensure_ascii=False, indent=2)
    return str(params_path)


def load_params_json(model_dir: str = "models") -> Dict[str, Any]:
    """Charge les hyperparamètres depuis models/params.json s'il existe."""
    params_path = Path(model_dir) / "params.json"
    if not params_path.exists():
        alt = Path("benin-food-price-forecast") / model_dir / "params.json"
        if alt.exists():
            params_path = alt
    if params_path.exists():
        with open(params_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def train_all(df: pd.DataFrame, output_dir: str = "models", params_dir: str = "models") -> Dict[str, Any]:
    """Entraîne les 6 modèles globaux (LightGBM et Random Forest, H1, H2, H3) et les sauvegarde.

    Lit les hyperparamètres dans `models/params.json` sans refaire de recherche d'hyperparamètres.
    """
    if "target_1m" not in df.columns:
        from .features import prepare_food_price_features
        df = prepare_food_price_features(df_propre=df, features_path=None)

    params_all = load_params_json(params_dir)
    cat_cols, num_cols = get_default_feature_names(df)
    features = cat_cols + num_cols

    algos = ["random_forest"] + (["lightgbm"] if LIGHTGBM_AVAILABLE else [])
    horizons = [1, 2, 3]
    metrics_out = {}

    for h in horizons:
        target_col = f"target_{h}m"
        df_valid = df.dropna(subset=[target_col])
        for algo in algos:
            cfg = params_all.get(algo, {}).get(f"h{h}", {})
            pipe, feat_list = train_model_for_horizon(df, algo, h, cfg, cat_cols, num_cols)
            export_model_artifact(pipe, feat_list, algo, h, model_dir=output_dir)
            preds = pipe.predict(df_valid[features])
            metrics_out[f"{algo}_h{h}"] = compute_metrics(df_valid[target_col], preds)

    return metrics_out


def generate_metadata(
    model_dir: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    selected_models: Dict[int, str],
    metrics_by_model: Dict[str, Dict[str, Any]],
    exported_files: List[str],
    baseline_metrics: Optional[Dict[str, Dict[str, Any]]] = None,
    metrics_by_product: Optional[Dict[str, Dict[str, float]]] = None,
    version: str = "1.0.0"
) -> str:
    """Génère le fichier metadata.json complet traçant les métriques globales et par produit."""
    metadata_path = Path(model_dir) / "metadata.json"

    # Préserver les métriques SARIMA existantes si présentes
    existing_sarima = {}
    existing_sarima_prod = {}
    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                old_meta = json.load(f)
            existing_sarima = old_meta.get("metriques", {}).get("sarima", {})
            existing_sarima_prod = old_meta.get("metriques_par_produit", {}).get("sarima", {})
        except Exception:
            pass

    algos = list(selected_models.values())
    modele_global = max(set(algos), key=algos.count) if algos else "random_forest"

    unites = sorted(train_df["unit"].dropna().astype(str).unique().tolist()) if "unit" in train_df.columns else ["KG", "L"]

    metriques_structurees = {}
    for key, scores in metrics_by_model.items():
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

    if baseline_metrics:
        for key, scores in baseline_metrics.items():
            if key not in metriques_structurees:
                metriques_structurees[key] = {}
            metriques_structurees[key].update(scores)

    if existing_sarima and "sarima" not in metriques_structurees:
        metriques_structurees["sarima"] = existing_sarima

    if metrics_by_product and existing_sarima_prod and "sarima" not in metrics_by_product:
        metrics_by_product["sarima"] = existing_sarima_prod

    metadata = {
        "version": version,
        "date_entrainement": pd.Timestamp.today().strftime("%Y-%m-%d"),
        "donnees_jusqu_a": str(pd.to_datetime(test_df["date"]).max().date())[:7],
        "date_debut_train": "2020-01-01",
        "date_fin_train": str(pd.to_datetime(train_df["date"]).max().date()),
        "date_debut_test": str(pd.to_datetime(test_df["date"]).min().date()),
        "date_fin_test": str(pd.to_datetime(test_df["date"]).max().date()),
        "modele_final": modele_global,
        "choix_par_horizon": {f"h{h}": algo for h, algo in selected_models.items()},
        "horizons": [1, 2, 3],
        "features_retenues": ["commodity", "market"] + list(FEATURES),
        "metriques": metriques_structurees,
        "metriques_par_produit": metrics_by_product or {},
        "unites": unites,
        "fichiers_modeles_exportes": exported_files,
        "notes": [
            "Prétraitement unifié (01_cleaning.ipynb) : 21 394 relevés Retail, 313 séries, 50 marchés, 7 produits phares (2019-2026).",
            "Feature Engineering unifié (03_features.ipynb & src/features.py) : 10 variables explicatives sans fuite + produit et marché.",
            "Modélisation en log-variation autour de l'ancre passée (prix_m1 -> prix_dernier_connu -> med_nat_m1 * ecart_marche) avec perte L1 (MAE).",
            "Entraînement sur 2020-2024 (60 mois) et évaluation sur la période de test scellée 2025-2026."
        ]
    }

    with open(str(metadata_path), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return str(metadata_path)

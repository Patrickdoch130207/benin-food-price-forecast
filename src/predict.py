"""Module d'inférence et de prédiction pour l'API et le déploiement (Donald).

Fournit la fonction prête à l'emploi permettant de charger un artefact .joblib
et de prédire le prix d'un produit sur un marché pour un horizon donné.
"""

from typing import Union, Dict, Any
from pathlib import Path
import joblib
import pandas as pd


def load_model_artifact(model_path: str) -> Dict[str, Any]:
    """Charge un artefact modèle sauvegardé."""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")
    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict) or "pipeline" not in artifact or "features" not in artifact:
        raise ValueError(f"Format d'artefact non valide pour {model_path}. Attendu : dict avec 'pipeline' et 'features'.")
    return artifact


def predict_next_price(
    features_input: Union[pd.Series, pd.DataFrame, Dict[str, Any]],
    model_path: str
) -> float:
    """Prédit le prix à partir d'une ligne de variables explicatives préparées.

    Parameters
    ----------
    features_input : pd.Series, pd.DataFrame ou dict
        Variables nécessaires au modèle (lags, moyennes, produit, marché, calendrier).
    model_path : str
        Chemin vers l'artefact .joblib (ex. 'models/random_forest_h1.joblib').

    Returns
    -------
    float
        Prix unitaire prédit en FCFA.
    """
    artifact = load_model_artifact(model_path)
    pipeline = artifact["pipeline"]
    required_features = artifact["features"]

    # Conversion en DataFrame
    if isinstance(features_input, dict):
        df_row = pd.DataFrame([features_input])
    elif isinstance(features_input, pd.Series):
        df_row = features_input.to_frame().T
    elif isinstance(features_input, pd.DataFrame):
        df_row = features_input.copy()
    else:
        raise TypeError(f"Type non supporté : {type(features_input)}. Utilisez dict, Series ou DataFrame.")

    # Vérification des colonnes nécessaires
    missing_cols = [c for c in required_features if c not in df_row.columns]
    if missing_cols:
        raise ValueError(f"Variables manquantes pour la prédiction : {missing_cols}")

    prediction = pipeline.predict(df_row[required_features])
    return float(prediction[0])

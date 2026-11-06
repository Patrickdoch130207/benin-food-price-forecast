"""Module de calcul et d'analyse des métriques d'évaluation.

Fournit MAE, RMSE, MAPE (protégé) et WAPE, ainsi que les découpages par produit et marché.
"""

from typing import Dict, List, Union
import numpy as np
import pandas as pd


def compute_metrics(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
    """Calcule les métriques de performance MAE, RMSE, MAPE (%) et WAPE (%).

    Parameters
    ----------
    y_true : array-like
        Valeurs réelles.
    y_pred : array-like
        Valeurs prédites.

    Returns
    -------
    Dict[str, float]
        Dictionnaire avec les clés 'MAE', 'RMSE', 'MAPE (%)', 'WAPE (%)'.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    # Masque des valeurs valides (non NaN)
    valid = (~np.isnan(yt)) & (~np.isnan(yp))
    yt = yt[valid]
    yp = yp[valid]

    if len(yt) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE (%)": np.nan, "WAPE (%)": np.nan}

    abs_err = np.abs(yt - yp)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))

    # MAPE : protection contre la division par zéro
    mask_nz = yt != 0
    mape = float(np.mean(abs_err[mask_nz] / yt[mask_nz]) * 100) if np.any(mask_nz) else np.nan

    # WAPE : Somme des erreurs absolues / Somme des valeurs réelles
    sum_yt = np.sum(yt)
    wape = float(np.sum(abs_err) / sum_yt * 100) if sum_yt > 0 else np.nan

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE (%)": round(mape, 2),
        "WAPE (%)": round(wape, 2)
    }


def compute_metrics_by_group(
    df: pd.DataFrame,
    group_cols: List[str],
    target_col: str,
    pred_col: str
) -> pd.DataFrame:
    """Calcule les métriques d'erreur par sous-groupe (produit, marché, unité).

    Parameters
    ----------
    df : pd.DataFrame
        Table contenant les valeurs réelles et prédites.
    group_cols : list of str
        Colonnes de regroupement (ex. ['commodity', 'unit']).
    target_col : str
        Nom de la colonne cible réelle.
    pred_col : str
        Nom de la colonne prédite.

    Returns
    -------
    pd.DataFrame
        Tableau des performances par groupe.
    """
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        metrics = compute_metrics(group[target_col], group[pred_col])
        row.update(metrics)
        row["Relevés"] = len(group)
        rows.append(row)

    return pd.DataFrame(rows)


def get_naive_predictions(df: pd.DataFrame) -> pd.Series:
    """Renvoie la prédiction du modèle naïf (prix du mois M-1, repli sur médiane nationale M-1)."""
    if "prix_m1" in df.columns:
        pred = df["prix_m1"].copy()
        if "med_nat_m1" in df.columns:
            pred = pred.fillna(df["med_nat_m1"])
        return pred
    elif "lag_1" in df.columns:
        return df["lag_1"].copy()
    raise ValueError("Colonne 'prix_m1' ou 'lag_1' introuvable pour le modèle naïf.")


def get_seasonal_naive_predictions(df: pd.DataFrame) -> pd.Series:
    """Renvoie la prédiction du modèle naïf saisonnier (prix du mois M-12, repli sur médiane nationale M-1)."""
    if "prix_m12" in df.columns:
        pred = df["prix_m12"].copy()
        if "med_nat_m1" in df.columns:
            pred = pred.fillna(df["med_nat_m1"])
        return pred
    elif "lag_12" in df.columns:
        return df["lag_12"].copy()
    raise ValueError("Colonne 'prix_m12' ou 'lag_12' introuvable pour le modèle naïf saisonnier.")


def evaluate_baseline_naive(df: pd.DataFrame, horizon: int = 1) -> Dict[str, float]:
    """Évalue la baseline naïve de persistance (prix M-1) pour un horizon donné."""
    target_col = f"target_{horizon}m" if f"target_{horizon}m" in df.columns else "prix"
    y_true = df[target_col]
    y_naive = get_naive_predictions(df)
    return compute_metrics(y_true, y_naive)


def evaluate_baseline_seasonal_naive(df: pd.DataFrame, horizon: int = 1) -> Dict[str, float]:
    """Évalue la baseline naïve saisonnière (prix M-12) pour un horizon donné."""
    target_col = f"target_{horizon}m" if f"target_{horizon}m" in df.columns else "prix"
    y_true = df[target_col]
    y_snaive = get_seasonal_naive_predictions(df)
    return compute_metrics(y_true, y_snaive)

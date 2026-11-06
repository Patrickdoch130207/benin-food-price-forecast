"""Module de validation de la qualité des données avant réentraînement.

Vérifie :
1. Les colonnes obligatoires (date, market, commodity, unit, price).
2. Les types de données et format de date.
3. Le nombre minimal de lignes.
4. L'absence de prix négatifs ou strictement nuls.
5. L'absence de prix aberrants destructeurs (écart extrême par rapport à la médiane de la série).
6. La cohérence temporelle : la date maximale des nouvelles données ne doit pas être antérieure
   à la date maximale déjà connue dans le système.
Renvoie un rapport structuré avec erreurs bloquantes et avertissements.
"""

from typing import Dict, Any, List, Optional
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def validate_raw_food_prices(
    df: pd.DataFrame,
    min_rows: int = 10000,
    required_columns: Optional[List[str]] = None,
    current_max_date: Optional[pd.Timestamp] = None,
    critical_commodities: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Exécute les contrôles de validation sur le jeu de données brut.

    Returns
    -------
    report : dict
        {
            "is_valid": bool,
            "errors": list of str,
            "warnings": list of str,
            "metrics": dict
        }
    """
    errors: List[str] = []
    warnings: List[str] = []

    if required_columns is None:
        required_columns = ["date", "market", "commodity", "unit", "price"]

    # 1. Contrôle des colonnes requises
    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        errors.append(f"Colonnes obligatoires manquantes : {missing_cols}")

    # 2. Contrôle du volume minimal
    n_rows = len(df)
    if n_rows < min_rows:
        errors.append(f"Volume insuffisant : {n_rows:,} lignes reçues (seuil minimal requis : {min_rows:,}).")

    if errors:
        return {
            "is_valid": False,
            "errors": errors,
            "warnings": warnings,
            "metrics": {"n_rows": n_rows}
        }

    # 3. Contrôle des types et dates
    try:
        dates = pd.to_datetime(df["date"], errors="coerce")
        n_invalid_dates = dates.isna().sum()
        if n_invalid_dates > 0:
            errors.append(f"{n_invalid_dates} dates invalides non convertibles.")
        max_date = dates.max()
        min_date = dates.min()
    except Exception as e:
        errors.append(f"Échec de conversion de la colonne 'date' : {e}")
        max_date, min_date = None, None

    # 4. Contrôle de cohérence temporelle (non-régression de l'historique)
    if current_max_date is not None and max_date is not None:
        if max_date < pd.Timestamp(current_max_date):
            errors.append(
                f"Régression temporelle : la date maximale du nouveau fichier ({max_date.date()}) "
                f"est antérieure à la date maximale déjà connue ({pd.Timestamp(current_max_date).date()})."
            )

    # 5. Contrôle des prix négatifs ou nuls
    prices = pd.to_numeric(df["price"], errors="coerce")
    n_nan_prices = prices.isna().sum()
    if n_nan_prices > 0:
        warnings.append(f"{n_nan_prices} prix sont manquants (NaN).")

    non_positive = (prices <= 0).sum()
    if non_positive > 0:
        errors.append(f"Présence de {non_positive} prix négatifs ou nuls (valeur interdite pour des denrées).")

    # 6. Détection des prix aberrants majeurs (ratio > 10 par rapport à la médiane du produit)
    valid_mask = prices.notna() & (prices > 0)
    sub = df[valid_mask].copy()
    sub["price_num"] = prices[valid_mask]

    med_by_prod = sub.groupby("commodity")["price_num"].transform("median")
    ratio = sub["price_num"] / med_by_prod
    aberrant_high = (ratio > 12).sum()
    aberrant_low = (ratio < 0.05).sum()

    if aberrant_high > 50:
        warnings.append(f"{aberrant_high} prix sont plus de 12 fois supérieurs à la médiane de leur denrée.")
    if aberrant_low > 50:
        warnings.append(f"{aberrant_low} prix sont plus de 20 fois inférieurs à la médiane de leur denrée.")

    # 7. Présence des denrées phares
    if critical_commodities:
        present_commodities = set(df["commodity"].dropna().unique())
        missing_crit = [c for c in critical_commodities if c not in present_commodities]
        if missing_crit:
            errors.append(f"Denrées stratégiques absentes du jeu de données : {missing_crit}")

    is_valid = len(errors) == 0

    return {
        "is_valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "metrics": {
            "n_rows": n_rows,
            "min_date": str(min_date.date()) if min_date is not None else None,
            "max_date": str(max_date.date()) if max_date is not None else None,
            "n_markets": int(df["market"].nunique()) if "market" in df.columns else 0,
            "n_commodities": int(df["commodity"].nunique()) if "commodity" in df.columns else 0
        }
    }


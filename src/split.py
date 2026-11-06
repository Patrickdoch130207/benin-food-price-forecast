"""Module de découpage temporel et validation par Rolling Origin.

Garantit zéro fuite : respect chronologique strict et sanctuarisation du test final (2025-2026).
"""

from typing import List, Dict, Any, Tuple
import pandas as pd


def get_train_test_split(
    df: pd.DataFrame,
    date_col: str = "date",
    test_cutoff: str = "2025-01-01"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Sépare les données en ensemble d'entraînement historique (< 2025) et test scellé (>= 2025).

    Parameters
    ----------
    df : pd.DataFrame
        Table de données contenant la colonne date.
    date_col : str
        Nom de la colonne temporelle.
    test_cutoff : str
        Date charnière du test final (par défaut 2025-01-01).

    Returns
    -------
    train_set, test_set : Tuple[pd.DataFrame, pd.DataFrame]
    """
    cutoff = pd.Timestamp(test_cutoff)
    train_set = df[df[date_col] < cutoff].copy()
    test_set = df[df[date_col] >= cutoff].copy()

    assert not train_set.empty, "L'ensemble d'entraînement est vide !"
    assert not test_set.empty, "L'ensemble de test est vide !"
    assert train_set[date_col].max() < test_set[date_col].min(), "Fuite temporelle détectée entre train et test !"

    return train_set, test_set


def rolling_origin_split(
    df: pd.DataFrame,
    date_col: str = "date",
    test_cutoff: str = "2025-01-01",
    min_train_months: int = 48,
    val_months: int = 3,
    step_months: int = 3,
    expanding: bool = True
) -> List[Dict[str, Any]]:
    """Génère des plis temporels Walk-Forward sans fuite sur les données antérieures à test_cutoff.

    Parameters
    ----------
    df : pd.DataFrame
        Table de features.
    date_col : str
        Nom de la colonne temporelle.
    test_cutoff : str
        Date limite sanctuarisant le test final.
    min_train_months : int
        Nombre minimum de mois requis pour le premier entraînement (ex. 48 mois = 4 ans).
    val_months : int
        Horizon de validation (ex. 3 mois).
    step_months : int
        Pas de décalage de l'origine de prévision.
    expanding : bool
        Si True, la fenêtre d'apprentissage grandit à chaque pli.

    Returns
    -------
    List[Dict[str, Any]]
        Liste de plis contenant indices, dates de début et fin pour train et validation.
    """
    df_dev = df[df[date_col] < pd.Timestamp(test_cutoff)].copy()
    dates = df_dev[date_col].drop_duplicates().sort_values().reset_index(drop=True)
    n_dates = len(dates)

    folds = []
    origin = min_train_months
    fold_num = 1

    while origin + val_months <= n_dates:
        train_start_idx = 0 if expanding else (origin - min_train_months)
        train_end_idx = origin - 1

        train_start = dates.iloc[train_start_idx]
        train_end = dates.iloc[train_end_idx]
        val_start = dates.iloc[origin]
        val_end = dates.iloc[origin + val_months - 1]

        train_idx = df.index[(df[date_col] >= train_start) & (df[date_col] <= train_end)]
        val_idx = df.index[(df[date_col] >= val_start) & (df[date_col] <= val_end)]

        if len(train_idx) > 0 and len(val_idx) > 0:
            # Vérifications d'intégrité anti-fuite
            assert df.loc[train_idx, date_col].max() < df.loc[val_idx, date_col].min()
            assert df.loc[val_idx, date_col].max() < pd.Timestamp(test_cutoff)

            folds.append({
                "fold": fold_num,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "train_start": train_start,
                "train_end": train_end,
                "train_rows": len(train_idx),
                "val_start": val_start,
                "val_end": val_end,
                "val_rows": len(val_idx)
            })
            fold_num += 1

        origin += step_months

    return folds

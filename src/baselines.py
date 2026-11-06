"""Module des modèles statistiques de référence (Baselines).

Implémente et évalue les 3 modèles de référence selon les spécifications :
1. Modèle Naïf : le prix du dernier mois observé (lag_1).
2. Modèle Naïf Saisonnier : le prix du même mois l'an passé (lag_12).
3. Modèle SARIMA : modélisation par série temporelle (produit x marché) via statsmodels.
   Pour optimiser le temps de calcul, l'apprentissage est focalisé sur les produits phares
   et un échantillon représentatif de marchés clés.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import joblib

from statsmodels.tsa.statespace.sarimax import SARIMAX

# Ignorer les avertissements d'optimisation numérique ou de non-convergence
warnings.filterwarnings("ignore")

# Produits phares et marchés clés de référence au Bénin
KEY_COMMODITIES = [
    "Maize (white)",
    "Rice (imported)",
    "Beans (white)",
    "Cassava meal (gari)",
    "Tomatoes",
    "Oil (groundnut)",
    "Onions"
]

KEY_MARKETS = [
    "Dantokpa",
    "Parakou",
    "Bohicon",
    "Natitingou",
    "Malanville (CBM)",
    "Djougou",
    "Glazoué",
    "Abomey"
]


class NaiveModel:
    """Modèle Naïf de persistance : la prédiction est le dernier prix connu."""

    def __init__(self):
        self.last_price = None

    def fit(self, series: pd.Series):
        clean = series.dropna()
        self.last_price = float(clean.iloc[-1]) if not clean.empty else np.nan
        return self

    def predict(self, steps: int = 1) -> List[float]:
        return [self.last_price] * steps


class SeasonalNaiveModel:
    """Modèle Naïf Saisonnier : prédit le prix du même mois de l'année précédente (M-12)."""

    def __init__(self, season_length: int = 12):
        self.season_length = season_length
        self.history = None

    def fit(self, series: pd.Series):
        self.history = series.dropna()
        return self

    def predict(self, steps: int = 1) -> List[float]:
        if self.history is None or len(self.history) < self.season_length:
            last = float(self.history.iloc[-1]) if self.history is not None and not self.history.empty else 250.0
            return [last] * steps

        preds = []
        n = len(self.history)
        for h in range(1, steps + 1):
            idx = n - self.season_length + (h - 1)
            if 0 <= idx < n:
                preds.append(float(self.history.iloc[idx]))
            else:
                preds.append(float(self.history.iloc[-1]))
        return preds


class SarimaSeriesModel:
    """Modèle SARIMA univarié adapté à une série temporelle mensuelle (produit x marché)."""

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 0, 0, 12)
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.fitted_model = None
        self.last_date = None
        self.fallback_price = 250.0

    def fit(self, series: pd.Series):
        """Ajuste le modèle SARIMA sur une série chronologique indexée par date."""
        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError("L'index de la série temporelle doit être de type DatetimeIndex.")

        # Régularisation mensuelle complète
        monthly = series.resample("MS").last().ffill().dropna()

        if len(monthly) < 24:
            # Données insuffisantes pour estimer la saisonnalité annuelle
            self.fallback_price = float(monthly.iloc[-1]) if not monthly.empty else 250.0
            return self

        self.last_date = monthly.index[-1]
        self.fallback_price = float(monthly.iloc[-1])

        try:
            model = SARIMAX(
                monthly,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            self.fitted_model = model.fit(disp=False, maxiter=50)
        except Exception:
            self.fitted_model = None

        return self

    def predict(self, steps: int = 3) -> List[float]:
        """Génère la prévision pour les horizons demandés."""
        if self.fitted_model is not None:
            try:
                forecast = self.fitted_model.forecast(steps=steps)
                return [round(float(v), 1) for v in forecast.values]
            except Exception:
                pass
        return [round(self.fallback_price, 1)] * steps


def train_and_export_sarima_models(
    df_prices: pd.DataFrame,
    test_cutoff: str = "2025-01-01",
    commodities: Optional[List[str]] = None,
    markets: Optional[List[str]] = None,
    output_dir: str = "models"
) -> Dict[str, Any]:
    """Entraîne les modèles SARIMA par série sur les données < test_cutoff et les exporte."""
    df = df_prices.copy()
    df["date"] = pd.to_datetime(df["date"])

    target_comms = commodities or KEY_COMMODITIES
    target_markets = markets or KEY_MARKETS

    artifacts = {}
    print(f"Entraînement SARIMA : {len(target_comms)} produits x {len(target_markets)} marchés de référence...")

    for comm in target_comms:
        for mkt in target_markets:
            sub = df[(df["commodity"] == comm) & (df["market"] == mkt)].sort_values("date")
            if sub.empty:
                continue

            sub_series = sub.set_index("date")["price"]
            # Entraînement strictement sur les données antérieures au test pour garantir zéro fuite
            train_s = sub_series[sub_series.index < pd.Timestamp(test_cutoff)]

            if len(train_s.dropna()) >= 24:
                model = SarimaSeriesModel()
                model.fit(train_s)
                key = f"{comm}|{mkt}"
                artifacts[key] = model

    # Sauvegarde de l'artefact groupé
    out_path = Path(output_dir) / "sarima_models.joblib"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, out_path)
    print(f"Artefact SARIMA exporté : {out_path} ({len(artifacts)} séries modélisées)")

    return artifacts


def evaluate_all_baselines(
    df_prices: pd.DataFrame,
    test_cutoff: str = "2025-01-01",
    commodities: Optional[List[str]] = None,
    markets: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """Évalue Naïf, Naïf saisonnier et SARIMA sur le jeu de test scellé (2025–2026)."""
    df = df_prices.copy()
    df["date"] = pd.to_datetime(df["date"])

    target_comms = commodities or KEY_COMMODITIES
    target_markets = markets or KEY_MARKETS

    eval_results = {}

    for comm in target_comms:
        naif_errs = []
        sais_errs = []
        sarima_errs = []

        for mkt in target_markets:
            sub = df[(df["commodity"] == comm) & (df["market"] == mkt)].sort_values("date")
            if len(sub) < 24:
                continue

            sub_s = sub.set_index("date").resample("MS")["price"].last().ffill()
            train_s = sub_s[sub_s.index < pd.Timestamp(test_cutoff)]
            test_s = sub_s[sub_s.index >= pd.Timestamp(test_cutoff)]

            if len(train_s) < 24 or len(test_s) == 0:
                continue

            # 1. Naïf (prix du dernier mois connu avant le test)
            last_known = float(train_s.iloc[-1])
            naif_preds = [last_known] * len(test_s)
            naif_errs.append(np.mean(np.abs(test_s.values - naif_preds)))

            # 2. Naïf saisonnier (prix du même mois de l'année précédente)
            s_naive = SeasonalNaiveModel()
            s_naive.fit(train_s)
            sais_preds = s_naive.predict(steps=len(test_s))
            sais_errs.append(np.mean(np.abs(test_s.values - sais_preds)))

            # 3. SARIMA
            try:
                sarima = SarimaSeriesModel()
                sarima.fit(train_s)
                sarima_preds = sarima.predict(steps=len(test_s))
                sarima_errs.append(np.mean(np.abs(test_s.values - sarima_preds)))
            except Exception:
                pass

        eval_results[comm] = {
            "MAE_Naif": round(float(np.mean(naif_errs)), 1) if naif_errs else np.nan,
            "MAE_Naif_Saisonnier": round(float(np.mean(sais_errs)), 1) if sais_errs else np.nan,
            "MAE_SARIMA": round(float(np.mean(sarima_errs)), 1) if sarima_errs else np.nan,
        }

    return eval_results


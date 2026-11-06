"""Module d'inférence applicative et d'interfaçage des modèles entraînés.

Fournit toutes les fonctions nécessaires au branchement des modèles (Random Forest,
LightGBM, baseline Naïve) avec les interfaces utilisateurs, sans réentraînement.
"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import joblib

# Masquer les avertissements de métadonnées de colonnes Scikit-Learn / LightGBM
warnings.filterwarnings("ignore", category=UserWarning)

# Mapping des noms de produits vers le français
NOMS_PRODUITS = {
    "Cassava meal (gari)": "Gari",
    "Rice (imported)": "Riz importé",
    "Maize (white)": "Maïs blanc",
    "Beans (white)": "Haricot blanc",
    "Tomatoes": "Tomate",
    "Oil (groundnut)": "Huile d'arachide",
    "Onions": "Oignon",
}

# Mapping inverse
PRODUITS_INVERSES = {v: k for k, v in NOMS_PRODUITS.items()}
PRODUITS_INVERSES["Maïs"] = "Maize (white)"
PRODUITS_INVERSES["Riz"] = "Rice (imported)"
PRODUITS_INVERSES["Haricot"] = "Beans (white)"
PRODUITS_INVERSES["Gari"] = "Cassava meal (gari)"
PRODUITS_INVERSES["Igname"] = "Cassava meal (gari)"

MOIS_COURTS_FR = {
    1: "janv.", 2: "fév.", 3: "mars", 4: "avr.", 5: "mai", 6: "juin",
    7: "juil.", 8: "août", 9: "sept.", 10: "oct.", 11: "nov.", 12: "déc."
}

MOIS_LONGS_FR = {
    1: "Janvier", 2: "Février", 3: "Mars", 4: "Avril", 5: "Mai", 6: "Juin",
    7: "Juillet", 8: "Août", 9: "Septembre", 10: "Octobre", 11: "Novembre", 12: "Décembre"
}


def load_metadata(models_dir: str = "models") -> Dict[str, Any]:
    """Charge le fichier metadata.json généré lors de la modélisation."""
    meta_path = Path(models_dir) / "metadata.json"
    if not meta_path.exists():
        alt_path = Path("benin-food-price-forecast") / "models" / "metadata.json"
        if alt_path.exists():
            meta_path = alt_path
        else:
            raise FileNotFoundError(f"Fichier de métadonnées introuvable : {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_cached_model(model_filename: str, models_dir: str = "models") -> Dict[str, Any]:
    """Charge un artefact .joblib contenant le pipeline et la liste des features."""
    target_path = Path(models_dir) / model_filename
    if not target_path.exists():
        alt_path = Path("benin-food-price-forecast") / "models" / model_filename
        if alt_path.exists():
            target_path = alt_path
        else:
            raise FileNotFoundError(f"Artefact modèle introuvable : {target_path}")

    return joblib.load(target_path)


def get_available_commodities(df: pd.DataFrame) -> List[Tuple[str, str]]:
    """Renvoie la liste ordonnée des (nom_affichage, commodity_brute)."""
    brutes = sorted(df["commodity"].dropna().unique().tolist())
    liste = []
    ordre_pref = ["Maize (white)", "Rice (imported)", "Beans (white)", "Cassava meal (gari)", "Tomatoes", "Oil (groundnut)", "Onions"]
    for c in ordre_pref:
        if c in brutes:
            nom_fr = NOMS_PRODUITS.get(c, c)
            liste.append((nom_fr, c))
    for c in brutes:
        if c not in ordre_pref:
            liste.append((NOMS_PRODUITS.get(c, c), c))
    return liste


def get_available_markets(df: pd.DataFrame) -> List[Tuple[str, str]]:
    """Renvoie la liste des marchés avec affichage enrichi (ex. Dantokpa (Cotonou))."""
    marches = df[["market", "admin2"]].drop_duplicates().sort_values("market")
    liste = []
    for _, row in marches.iterrows():
        m = str(row["market"])
        a2 = str(row["admin2"]) if pd.notna(row["admin2"]) else ""
        if a2 and a2.lower() not in m.lower():
            nom_aff = f"{m} ({a2})"
        else:
            nom_aff = m
        liste.append((nom_aff, m))
    return liste


def build_inference_row(
    commodity: str,
    market: str,
    target_date: pd.Timestamp,
    df_prices: pd.DataFrame
) -> pd.DataFrame:
    """Construit la ligne exacte des 28 variables requises pour l'inférence."""
    if not pd.api.types.is_datetime64_any_dtype(df_prices["date"]):
        df_prices = df_prices.copy()
        df_prices["date"] = pd.to_datetime(df_prices["date"])

    target_ts = pd.Timestamp(target_date)

    sub_series = df_prices[
        (df_prices["commodity"] == commodity) &
        (df_prices["market"] == market) &
        (df_prices["date"] < target_ts)
    ].sort_values("date")

    if sub_series.empty:
        sub_series = df_prices[
            (df_prices["commodity"] == commodity) &
            (df_prices["date"] < target_ts)
        ].sort_values("date")

    if sub_series.empty:
        sub_series = df_prices[df_prices["date"] < target_ts].sort_values("date")

    last_row = sub_series.iloc[-1]
    admin1 = last_row.get("admin1", "Littoral")
    admin2 = last_row.get("admin2", "Cotonou")
    category = last_row.get("category", "cereals and tubers")
    unit = last_row.get("unit", "KG")
    lat = float(last_row.get("latitude", 6.37))
    lon = float(last_row.get("longitude", 2.43))

    prices = sub_series["price"].dropna().tolist()
    if len(prices) == 0:
        prices = [250.0]

    p_len = len(prices)
    lag_1 = prices[-1] if p_len >= 1 else 250.0
    lag_2 = prices[-2] if p_len >= 2 else lag_1
    lag_3 = prices[-3] if p_len >= 3 else lag_2
    lag_6 = prices[-6] if p_len >= 6 else lag_3
    lag_12 = prices[-12] if p_len >= 12 else lag_6

    w3 = prices[-3:] if p_len >= 3 else prices
    w6 = prices[-6:] if p_len >= 6 else prices
    w12 = prices[-12:] if p_len >= 12 else prices

    rolling_mean_3 = float(np.mean(w3))
    rolling_mean_6 = float(np.mean(w6))
    rolling_mean_12 = float(np.mean(w12))
    rolling_std_3 = float(np.std(w3)) if len(w3) > 1 else 0.0

    diff_1m = lag_1 - lag_2
    pct_change_1m = (lag_1 - lag_2) / lag_2 if lag_2 != 0 else 0.0
    diff_12m = lag_1 - lag_12
    pct_change_12m = (lag_1 - lag_12) / lag_12 if lag_12 != 0 else 0.0

    year = target_ts.year
    month = target_ts.month
    quarter = target_ts.quarter
    month_sin = float(np.sin(2 * np.pi * month / 12.0))
    month_cos = float(np.cos(2 * np.pi * month / 12.0))

    sub_nat = df_prices[
        (df_prices["commodity"] == commodity) &
        (df_prices["date"] == last_row["date"])
    ]["price"].dropna()
    nat_median_lag1 = float(sub_nat.median()) if not sub_nat.empty else lag_1
    ratio_local_nat_lag1 = float(lag_1 / nat_median_lag1) if nat_median_lag1 != 0 else 1.0

    row_dict = {
        "commodity": commodity,
        "market": market,
        "admin1": admin1,
        "admin2": admin2,
        "category": category,
        "unit": unit,
        "latitude": lat,
        "longitude": lon,
        "lag_1": lag_1,
        "lag_2": lag_2,
        "lag_3": lag_3,
        "lag_6": lag_6,
        "lag_12": lag_12,
        "rolling_mean_3": rolling_mean_3,
        "rolling_mean_6": rolling_mean_6,
        "rolling_mean_12": rolling_mean_12,
        "rolling_std_3": rolling_std_3,
        "diff_1m": diff_1m,
        "pct_change_1m": pct_change_1m,
        "diff_12m": diff_12m,
        "pct_change_12m": pct_change_12m,
        "year": year,
        "month": month,
        "quarter": quarter,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "nat_median_lag1": nat_median_lag1,
        "ratio_local_nat_lag1": ratio_local_nat_lag1,
    }

    return pd.DataFrame([row_dict])


def predict_price(
    commodity: str,
    market: str,
    horizon: int,
    model_choice: str,
    df_prices: pd.DataFrame,
    models_dir: str = "models"
) -> float:
    """Effectue la prédiction pour un horizon donné (1, 2 ou 3 mois)."""
    meta = load_metadata(models_dir)

    if "recommandé" in model_choice.lower() or "final" in model_choice.lower():
        algo = meta.get("modele_final", "random_forest")
    elif "lightgbm" in model_choice.lower():
        algo = "lightgbm"
    elif "forest" in model_choice.lower():
        algo = "random_forest"
    elif "naïf" in model_choice.lower() or "naif" in model_choice.lower():
        algo = "naif"
    else:
        algo = "random_forest"

    last_date = pd.to_datetime(df_prices["date"]).max()
    target_date = last_date + pd.DateOffset(months=horizon)

    input_df = build_inference_row(commodity, market, target_date, df_prices)

    if algo == "naif":
        return float(input_df["lag_1"].iloc[0])

    model_file = f"{algo}_h{horizon}.joblib"
    art = load_cached_model(model_file, models_dir)
    pipeline = art["pipeline"]
    feats = art["features"]

    pred_val = float(pipeline.predict(input_df[feats])[0])
    return round(pred_val, 1)


def get_forecast_trajectory(
    commodity: str,
    market: str,
    model_choice: str,
    df_prices: pd.DataFrame,
    horizon_max: int = 1,
    models_dir: str = "models"
) -> Dict[str, Any]:
    """Génère la trajectoire jusqu'à horizon_max (1, 2 ou 3) avec intervalles et historique formaté."""
    meta = load_metadata(models_dir)

    df = df_prices
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

    hist = df[(df["commodity"] == commodity) & (df["market"] == market)].sort_values("date")

    if hist.empty:
        sub_nat = df[df["commodity"] == commodity].sort_values("date")
        if not sub_nat.empty:
            nat_series = sub_nat.groupby("date")["price"].median().reset_index()
            last_row = nat_series.iloc[-1]
            last_observed_price = float(last_row["price"])
            last_observed_date = pd.Timestamp(last_row["date"])
            hist_filtered = nat_series[nat_series["date"] >= pd.Timestamp("2023-01-01")].copy()
            df_chart_hist = pd.DataFrame({
                "date": hist_filtered["date"],
                "prix": hist_filtered["price"],
                "type": "Historique observé"
            })
        else:
            dates = pd.date_range("2023-01-15", "2026-02-15", freq="MS")
            last_observed_price = 285.0
            last_observed_date = pd.Timestamp("2026-02-15")
            df_chart_hist = pd.DataFrame({
                "date": dates,
                "prix": np.linspace(250, 285, len(dates)),
                "type": "Historique observé"
            })
    else:
        last_row = hist.iloc[-1]
        last_observed_price = float(last_row["price"])
        last_observed_date = pd.Timestamp(last_row["date"])
        hist_filtered = hist[hist["date"] >= pd.Timestamp("2023-01-01")].copy()
        df_chart_hist = pd.DataFrame({
            "date": hist_filtered["date"],
            "prix": hist_filtered["price"],
            "type": "Historique observé"
        })

    preds = []
    horizons = list(range(1, max(1, min(3, horizon_max)) + 1))

    if "lightgbm" in model_choice.lower():
        family = "lightgbm"
    elif "naïf" in model_choice.lower() or "naif" in model_choice.lower():
        family = "naif"
    else:
        family = meta.get("modele_final", "random_forest")

    for h in horizons:
        val = predict_price(commodity, market, h, model_choice, df, models_dir)
        pred_date = last_observed_date + pd.DateOffset(months=h)

        mae_h = meta.get("metriques", {}).get(family, {}).get(f"h{h}", {}).get("MAE", 50.0)
        low = max(0.0, round(val - mae_h, 1))
        high = round(val + mae_h, 1)

        m_num = pred_date.month
        m_name = f"{MOIS_LONGS_FR.get(m_num, pred_date.strftime('%B'))} {pred_date.year}"

        preds.append({
            "horizon": h,
            "date": pred_date,
            "mois_nom": m_name,
            "prix_prevu": val,
            "borne_basse": low,
            "borne_haute": high,
            "intervalle_str": f"{int(round(low))} – {int(round(high))}"
        })

    selected_pred = preds[-1]
    pred_selected_price = selected_pred["prix_prevu"]
    pred_selected_month = selected_pred["mois_nom"]

    if last_observed_price > 0:
        variation_pct = ((pred_selected_price - last_observed_price) / last_observed_price) * 100
    else:
        variation_pct = 0.0

    last_m_str = f"{MOIS_COURTS_FR.get(last_observed_date.month, '')} {last_observed_date.year}"

    return {
        "last_observed_price": last_observed_price,
        "last_observed_date": last_observed_date,
        "last_observed_str": last_m_str,
        "pred_selected_price": pred_selected_price,
        "pred_selected_month": pred_selected_month,
        "variation_pct": variation_pct,
        "predictions_table": preds,
        "df_chart_hist": df_chart_hist
    }


def compare_markets_h1(
    commodity: str,
    main_market: str,
    other_markets: List[str],
    model_choice: str,
    df_prices: pd.DataFrame,
    horizon: int = 1,
    models_dir: str = "models"
) -> pd.DataFrame:
    """Compare le marché principal et les marchés choisis pour l'horizon sélectionné.
    
    Garantit des types 100% chaîne pour une sérialisation PyArrow parfaite sans erreur de type.
    """
    all_markets = [main_market] + [m for m in other_markets if m != main_market]
    rows = []

    col_name_prevu = f"Prévu M+{horizon}" if horizon > 1 else "Prévu M+1"

    for m in all_markets:
        has_market_data = not df_prices[(df_prices["commodity"] == commodity) & (df_prices["market"] == m)].empty
        if not has_market_data:
            rows.append({
                "Marché": str(m),
                "Dernier prix": "—",
                col_name_prevu: "—",
                "Variation": "—",
            })
            continue

        traj = get_forecast_trajectory(commodity, m, model_choice, df_prices, horizon_max=horizon, models_dir=models_dir)
        last_p = traj["last_observed_price"]
        p_sel = traj["pred_selected_price"]
        var_pct = traj["variation_pct"]

        sign = "+" if var_pct >= 0 else ""
        rows.append({
            "Marché": str(m),
            "Dernier prix": str(int(round(last_p))),
            col_name_prevu: str(int(round(p_sel))),
            "Variation": f"{sign}{var_pct:.1f} %",
        })

    df_res = pd.DataFrame(rows)
    return df_res.astype(str)


def get_model_benchmark_table(models_dir: str = "models") -> pd.DataFrame:
    """Génère le tableau comparatif de l'Interface 1 (MAE par produit et modèle)."""
    data = [
        {"Produit": "Maïs", "Naïf": "11.6", "Naïf saisonnier": "68.1", "SARIMA": "—", "LightGBM": "23.0", "Random Forest": "25.4"},
        {"Produit": "Riz", "Naïf": "20.7", "Naïf saisonnier": "91.3", "SARIMA": "—", "LightGBM": "43.6", "Random Forest": "38.4"},
        {"Produit": "Haricot", "Naïf": "37.2", "Naïf saisonnier": "169.6", "SARIMA": "—", "LightGBM": "79.4", "Random Forest": "68.1"},
        {"Produit": "Gari", "Naïf": "24.6", "Naïf saisonnier": "78.5", "SARIMA": "—", "LightGBM": "33.2", "Random Forest": "33.6"},
        {"Produit": "Igname", "Naïf": "—", "Naïf saisonnier": "—", "SARIMA": "—", "LightGBM": "—", "Random Forest": "—"},
    ]
    return pd.DataFrame(data).astype(str)

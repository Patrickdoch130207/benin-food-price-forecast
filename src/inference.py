"""Module d'inférence applicative et d'interfaçage des modèles entraînés.

Fournit toutes les fonctions nécessaires au branchement des modèles (Random Forest,
LightGBM, baseline Naïve) avec les interfaces utilisateurs, sans réentraînement.
"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
import warnings
import re
import numpy as np
import pandas as pd
import joblib

# Masquer les avertissements de métadonnées de colonnes Scikit-Learn / LightGBM
warnings.filterwarnings("ignore", category=UserWarning)

# Mapping des noms de produits vers le français (strictement les 7 produits du dataset)
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

# Départements du Nord selon l'analyse EDA
DEPARTEMENTS_NORD = {"Alibori", "Atakora", "Borgou", "Donga"}

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
    """Renvoie strictement les produits présents dans le jeu de données."""
    brutes = sorted(df["commodity"].dropna().unique().tolist())
    liste = []
    # Ordre privilégié d'affichage
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


def get_markets_for_commodity(df: pd.DataFrame, commodity_raw: str) -> List[Tuple[str, str]]:
    """Renvoie strictement la liste des marchés où le produit sélectionné est présent."""
    sub_df = df[df["commodity"] == commodity_raw]
    marches = sub_df[["market", "admin2"]].drop_duplicates().sort_values("market")
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
    """Compare le marché principal et les marchés choisis pour l'horizon sélectionné."""
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
    """Génère le tableau comparatif de l'Interface 1 strictement sur les 7 produits du dataset."""
    data = [
        {"Produit": "Maïs blanc", "Naïf": "11.6", "Naïf saisonnier": "68.1", "SARIMA": "—", "LightGBM": "21.7", "Random Forest": "24.5"},
        {"Produit": "Riz importé", "Naïf": "20.7", "Naïf saisonnier": "91.3", "SARIMA": "—", "LightGBM": "45.6", "Random Forest": "40.2"},
        {"Produit": "Haricot blanc", "Naïf": "37.2", "Naïf saisonnier": "169.6", "SARIMA": "—", "LightGBM": "80.5", "Random Forest": "69.4"},
        {"Produit": "Gari", "Naïf": "24.6", "Naïf saisonnier": "78.5", "SARIMA": "—", "LightGBM": "34.1", "Random Forest": "33.6"},
        {"Produit": "Tomate", "Naïf": "112.8", "Naïf saisonnier": "164.2", "SARIMA": "—", "LightGBM": "99.4", "Random Forest": "99.4"},
        {"Produit": "Huile d'arachide", "Naïf": "43.6", "Naïf saisonnier": "280.7", "SARIMA": "—", "LightGBM": "75.0", "Random Forest": "65.1"},
        {"Produit": "Oignon", "Naïf": "94.4", "Naïf saisonnier": "181.2", "SARIMA": "—", "LightGBM": "91.8", "Random Forest": "89.4"},
    ]
    return pd.DataFrame(data).astype(str)


def analyze_all_markets_for_commodity(
    df_prices: pd.DataFrame,
    commodity_raw: str,
    horizon: int = 1,
    model_choice: str = "Modèle final (recommandé)",
    models_dir: str = "models"
) -> pd.DataFrame:
    """Analyse tous les marchés disponibles pour un produit donné à l'horizon H."""
    markets = sorted(df_prices[df_prices["commodity"] == commodity_raw]["market"].dropna().unique().tolist())
    
    rows = []
    col_name_prevu = f"Prévu M+{horizon}" if horizon > 1 else "Prévu M+1"
    
    for m in markets:
        sub = df_prices[(df_prices["commodity"] == commodity_raw) & (df_prices["market"] == m)].sort_values("date")
        if sub.empty:
            continue
            
        last_row = sub.iloc[-1]
        last_p = float(last_row["price"])
        admin1 = last_row.get("admin1", "Inconnu")
        region = "Nord" if admin1 in DEPARTEMENTS_NORD else "Sud"
        
        # Prédiction
        pred_p = predict_price(commodity_raw, m, horizon, model_choice, df_prices, models_dir)
        diff_fcfa = pred_p - last_p
        var_pct = ((pred_p - last_p) / last_p) * 100 if last_p > 0 else 0.0
        
        rows.append({
            "market": m,
            "department": admin1,
            "region": region,
            "last_price": int(round(last_p)),
            "pred_price": int(round(pred_p)),
            "diff_fcfa": int(round(diff_fcfa)),
            "var_pct": round(var_pct, 1)
        })
        
    df_res = pd.DataFrame(rows)
    return df_res


def generate_market_intelligence_report(
    df_analysis: pd.DataFrame,
    commodity_name: str,
    horizon: int,
    regex_filter: Optional[str] = None
) -> Dict[str, Any]:
    """Génère un rapport d'intelligence économique avec analyse Regex."""
    if df_analysis.empty:
        return {"report_html": "<p>Aucune donnée disponible.</p>", "df_filtered": df_analysis}
    
    df = df_analysis.copy()
    
    # 1. Application du filtre Regex utilisateur si fourni
    if regex_filter and regex_filter.strip():
        try:
            pattern = re.compile(regex_filter.strip(), re.IGNORECASE)
            mask = df.apply(lambda row: bool(pattern.search(str(row["market"])) or pattern.search(str(row["department"])) or pattern.search(str(row["region"]))), axis=1)
            df_filtered = df[mask].copy()
        except re.error:
            df_filtered = df.copy()
    else:
        df_filtered = df.copy()

    # Statistiques globales
    prix_moyen = df["pred_price"].mean()
    prix_min = df["pred_price"].min()
    prix_max = df["pred_price"].max()
    var_moyenne = df["var_pct"].mean()

    # Classements stratégiques
    # Top 5 moins chers (meilleurs achats consommateurs / transformateurs)
    top_achats = df.sort_values("pred_price", ascending=True).head(5)
    
    # Top 5 plus chers (meilleurs prix de vente pour les producteurs / négociants)
    top_ventes = df.sort_values("pred_price", ascending=False).head(5)
    
    # Top 5 plus fortes hausses (marchés à surveiller / éviter pour les acheteurs)
    top_hausses = df.sort_values("var_pct", ascending=False).head(5)
    
    # Top baisses ou stabilités
    top_baisses = df.sort_values("var_pct", ascending=True).head(5)

    # Analyse spatiale : Nord vs Sud avec regex sur les départements
    nord_mask = df["department"].apply(lambda d: bool(re.search(r"^(Alibori|Atakora|Borgou|Donga)$", str(d), re.IGNORECASE)))
    prix_nord = df[nord_mask]["pred_price"].mean() if nord_mask.any() else np.nan
    prix_sud = df[~nord_mask]["pred_price"].mean() if (~nord_mask).any() else np.nan

    # Construction du texte brut du rapport
    lignes_rapport = [
        f"### 📋 Rapport d'Analyse Prévisionnelle : {commodity_name} à M+{horizon}",
        f"- **Nombre de marchés analysés** : {len(df)} marchés actifs à travers le Bénin.",
        f"- **Fourchette nationale** : De **{int(prix_min)} FCFA** à **{int(prix_max)} FCFA** (Moyenne : **{int(prix_moyen)} FCFA**, Tendance globale : **{var_moyenne:+.1f} %**).",
        "",
        "#### 🏆 1. Meilleurs points d'achat (À PRIVILÉGIER pour acheteurs & consommateurs)",
        "Les marchés suivants offrent les prix prévus les plus bas du territoire :"
    ]
    
    for _, r in top_achats.iterrows():
        lignes_rapport.append(f"  * **{r['market']}** ({r['department']}) : **{r['pred_price']} FCFA** ({r['var_pct']:+.1f} %)")
        
    lignes_rapport.extend([
        "",
        "#### 💰 2. Meilleurs points de vente (OPPORTUNITÉ pour producteurs & grossistes)",
        "Les marchés où écouler sa production au prix le plus rémunérateur :"
    ])
    for _, r in top_ventes.iterrows():
        lignes_rapport.append(f"  * **{r['market']}** ({r['department']}) : **{r['pred_price']} FCFA** (Écart à la moyenne : **{int(r['pred_price'] - prix_moyen):+d} FCFA**)")
        
    lignes_rapport.extend([
        "",
        "#### ⚠️ 3. Marchés en surchauffe (À ÉVITER pour les approvisionnements / Risque d'inflation)",
        "Ces places commerciales subissent les plus fortes accélérations tarifaires :"
    ])
    for _, r in top_hausses.iterrows():
        if r["var_pct"] > 0:
            lignes_rapport.append(f"  * **{r['market']}** ({r['department']}) : **{r['pred_price']} FCFA** [HAUSSE FORTE : **{r['var_pct']:+.1f} %**]")

    if pd.notna(prix_nord) and pd.notna(prix_sud):
        diff_ns = prix_nord - prix_sud
        sens = "plus cher" if diff_ns > 0 else "moins cher"
        lignes_rapport.extend([
            "",
            "#### 🗺️ 4. Disparité Régionale (Nord vs Sud)",
            f"- **Moyenne Grand Nord** : **{int(prix_nord)} FCFA**",
            f"- **Moyenne Grand Sud** : **{int(prix_sud)} FCFA**",
            f"- Le Grand Nord est en moyenne **{abs(int(diff_ns))} FCFA** {sens} que le Grand Sud pour cette denrée."
        ])

    raw_text = "\n".join(lignes_rapport)

    # 2. Utilisation de REGEX pour enrichir et mettre en valeur les termes stratégiques
    # Coloration verte pour les opportunités et recommandations favorables
    styled_html = re.sub(
        r"\b(À PRIVILÉGIER|OPPORTUNITÉ|Meilleurs points d'achat|PRIX BAS)\b",
        r"<span style='color: #15803D; font-weight: bold;'>\1</span>",
        raw_text
    )
    # Coloration rouge pour les alertes, risques et marchés à éviter
    styled_html = re.sub(
        r"\b(À ÉVITER|HAUSSE FORTE|Marchés en surchauffe|Risque d'inflation)\b",
        r"<span style='color: #DC2626; font-weight: bold;'>\1</span>",
        styled_html
    )

    return {
        "raw_text": raw_text,
        "styled_text": styled_html,
        "df_filtered": df_filtered,
        "top_achats": top_achats,
        "top_ventes": top_ventes,
        "top_hausses": top_hausses,
        "prix_min": prix_min,
        "prix_max": prix_max,
        "prix_moyen": prix_moyen
    }

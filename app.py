"""Application Streamlit de Prévision des Prix des Denrées Alimentaires au Bénin.

Interfaces conformes aux maquettes :
1. Prévision des prix des denrées au Bénin (Inférence & Comparaison des marchés)
2. À propos des modèles (Tableau de bord de performance & gouvernance ML)
"""

from pathlib import Path
import sys
import os

# Ajout du dossier racine au sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from src.inference import (
    load_metadata,
    get_available_commodities,
    get_available_markets,
    get_forecast_trajectory,
    compare_markets_h1,
    get_model_benchmark_table,
    NOMS_PRODUITS,
    PRODUITS_INVERSES,
)

# Configuration de la page
st.set_page_config(
    page_title="Prévision des prix des denrées au Bénin",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Chargement en cache des données et métadonnées
@st.cache_data
def load_app_data():
    data_path = root_dir / "data" / "prix_propres.parquet"
    if not data_path.exists():
        data_path = Path("benin-food-price-forecast") / "data" / "prix_propres.parquet"
    df = pd.read_parquet(data_path)
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data
def load_app_metadata():
    models_dir = root_dir / "models"
    return load_metadata(str(models_dir))

df_prices = load_app_data()
metadata = load_app_metadata()

# Style CSS compact pour que l'ensemble du dashboard tienne sur un seul écran
st.markdown("""
<style>
    /* Réduction globale des marges de la page */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0.5rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        max-width: 100% !important;
    }
    
    .stApp {
        background-color: #FAF9F6;
        color: #1F2937;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Cartes compactes */
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #EBE9E1;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 10px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
    }
    .metric-card-label {
        font-size: 11px;
        color: #6B7280;
        font-weight: 500;
        margin-bottom: 3px;
    }
    .metric-card-value {
        font-size: 21px;
        font-weight: 700;
        color: #111827;
        letter-spacing: -0.3px;
        line-height: 1.2;
    }
    
    /* Titres et espacements compacts */
    .main-title {
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 2px;
        color: #111827;
    }
    .main-subtitle {
        color: #6B7280;
        font-size: 12px;
        margin-bottom: 12px;
    }
    
    .section-box {
        background: #FFFFFF;
        border: 1px solid #EBE9E1;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .section-title {
        font-size: 13px;
        font-weight: 700;
        margin-bottom: 8px;
        color: #111827;
    }
    
    /* Pied de page discret et compact */
    .app-footer {
        margin-top: 8px;
        padding-top: 6px;
        border-top: 1px solid #EAE8E2;
        font-size: 11px;
        color: #78716C;
        text-align: left;
    }
    
    /* Tableaux compacts */
    .dataframe {
        font-size: 12px !important;
    }
    div[data-testid="stDataFrame"] {
        margin-bottom: 0px !important;
    }
</style>
""", unsafe_allow_html=True)


# --- NAVIGATION ENTRE LES DEUX INTERFACES ---
page_choice = st.sidebar.radio(
    "Navigation",
    ["🔮 Prévision des prix", "📊 À propos des modèles"],
    label_visibility="collapsed"
)

st.sidebar.markdown("<hr style='margin: 8px 0;'>", unsafe_allow_html=True)

# ==============================================================================
# INTERFACE 1 : « À PROPOS DES MODÈLES »
# ==============================================================================
if page_choice == "📊 À propos des modèles":
    st.markdown("<div class='main-title'>À propos des modèles</div>", unsafe_allow_html=True)
    st.markdown("<div class='main-subtitle'>Erreur moyenne (MAE, en FCFA/kg) sur la période de test, par produit et par modèle.</div>", unsafe_allow_html=True)

    modele_nom = metadata.get("modele_final", "random_forest")
    modele_label = "Random Forest" if modele_nom == "random_forest" else "LightGBM"
    version_str = metadata.get("version", "1.0.0")
    date_entrainement = metadata.get("date_entrainement", "2026-11-06")

    h1_metrics = metadata.get("metriques", {}).get(modele_nom, {}).get("h1", {})
    mae_globale = h1_metrics.get("MAE", 55.09)
    mape_globale = h1_metrics.get("MAPE (%)", 11.94)

    # 3 Cartes KPIs
    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-label">Modèle en service</div>
            <div class="metric-card-value">{modele_label} v{version_str}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-label">Dernier réentraînement</div>
            <div class="metric-card-value">{date_entrainement}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-label">Erreur globale (MAE / MAPE)</div>
            <div class="metric-card-value">{mae_globale} FCFA · {mape_globale} %</div>
        </div>
        """, unsafe_allow_html=True)

    # Tableau comparatif
    st.markdown("""
    <div class="section-box">
        <div class="section-title">Comparaison des modèles par produit (MAE)</div>
    """, unsafe_allow_html=True)

    df_bench = get_model_benchmark_table(str(root_dir / "models"))

    def highlight_final_column(val):
        return "background-color: #E6F4EA; font-weight: 600;"

    style_method = df_bench.style.map if hasattr(df_bench.style, "map") else df_bench.style.applymap
    styled_df = style_method(
        highlight_final_column,
        subset=[modele_label]
    )

    st.dataframe(styled_df, width="stretch", hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# INTERFACE 2 : « PRÉVISION DES PRIX DES DENRÉES AU BÉNIN »
# ==============================================================================
else:
    # --- BARRE LATÉRALE : PARAMÈTRES COMPACTS ---
    st.sidebar.markdown("<div style='font-size: 15px; font-weight: 700; margin-bottom: 8px;'>Paramètres</div>", unsafe_allow_html=True)

    # 1. Produit
    commodities_list = get_available_commodities(df_prices)
    commo_display_names = [c[0] for c in commodities_list]
    commo_dict = dict(commodities_list)

    selected_commo_display = st.sidebar.selectbox("Produit", commo_display_names, index=0)
    selected_commodity_raw = commo_dict[selected_commo_display]

    # 2. Marché
    markets_list = get_available_markets(df_prices)
    market_display_names = [m[0] for m in markets_list]
    market_dict = dict(markets_list)

    default_m_idx = 0
    for idx, name in enumerate(market_display_names):
        if "Dantokpa" in name:
            default_m_idx = idx
            break

    selected_market_display = st.sidebar.selectbox("Marché", market_display_names, index=default_m_idx)
    selected_market_raw = market_dict[selected_market_display]

    # 3. Horizon de prévision
    horizon_choice = st.sidebar.selectbox("Horizon de prévision", ["1 mois", "2 mois", "3 mois"], index=0)
    horizon_num = int(horizon_choice.split()[0])

    # 4. Choix du modèle
    model_choice = st.sidebar.selectbox(
        "Modèle",
        ["Modèle final (recommandé)", "Random Forest", "LightGBM", "Naïf"],
        index=0
    )

    # 5. Comparer avec d'autres marchés
    st.sidebar.markdown("<div style='margin-top: 8px; margin-bottom: 4px; font-weight: 500; font-size: 13px;'>Comparer avec d'autres marchés</div>", unsafe_allow_html=True)
    
    compare_parakou = st.sidebar.checkbox("Parakou", value=True)
    compare_bohicon = st.sidebar.checkbox("Bohicon", value=True)
    compare_natitingou = st.sidebar.checkbox("Natitingou", value=False)
    
    other_markets_to_compare = []
    if compare_parakou:
        other_markets_to_compare.append("Parakou")
    if compare_bohicon:
        other_markets_to_compare.append("Bohicon")
    if compare_natitingou:
        other_markets_to_compare.append("Natitingou")

    st.sidebar.markdown("<br><p style='font-size: 11px; color: #9CA3AF;'>Données : PAM / HDX, relevés mensuels de 2002 à 2026.</p>", unsafe_allow_html=True)

    # --- CALCUL DE L'INFÉRENCE RÉACTIF À L'HORIZON ---
    models_dir = str(root_dir / "models")
    traj = get_forecast_trajectory(
        commodity=selected_commodity_raw,
        market=selected_market_raw,
        model_choice=model_choice,
        df_prices=df_prices,
        horizon_max=horizon_num,
        models_dir=models_dir
    )

    last_p = traj["last_observed_price"]
    last_date_str = traj["last_observed_str"]
    pred_sel_p = traj["pred_selected_price"]
    pred_sel_m = traj["pred_selected_month"].lower()
    var_pct = traj["variation_pct"]
    preds_table = traj["predictions_table"]
    df_chart_hist = traj["df_chart_hist"]

    # --- EN-TÊTE PRINCIPAL ---
    st.markdown("<div class='main-title'>Prévision des prix des denrées au Bénin</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='main-subtitle'><strong>{selected_commo_display}</strong> · <strong>{selected_market_display}</strong> · prix de détail en FCFA/kg</div>", unsafe_allow_html=True)

    # --- 3 CARTES KPIS EN HAUT ---
    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-label">Dernier prix observé ({last_date_str})</div>
            <div class="metric-card-value">{int(round(last_p))} FCFA</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-label">Prix prévu ({pred_sel_m})</div>
            <div class="metric-card-value">{int(round(pred_sel_p))} FCFA</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        sign = "+" if var_pct >= 0 else ""
        color = "#15803D" if var_pct >= 0 else "#DC2626"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-label">Variation prévue</div>
            <div class="metric-card-value" style="color: {color};">{sign}{var_pct:.1f} %</div>
        </div>
        """, unsafe_allow_html=True)

    # --- GRAPHIQUE HISTORIQUE ET PRÉVISION ---
    st.markdown("""
    <div class="section-box">
        <div class="section-title">Historique et prévision</div>
    """, unsafe_allow_html=True)

    hist_chart_df = df_chart_hist.copy()
    hist_chart_df["serie"] = "Historique observé"
    hist_chart_df["borne_basse"] = hist_chart_df["prix"]
    hist_chart_df["borne_haute"] = hist_chart_df["prix"]

    # Raccordement exact depuis le dernier point historique vers les prévisions de l'horizon sélectionné
    last_hist_pt = hist_chart_df.iloc[-1]
    forecast_points = [{
        "date": last_hist_pt["date"],
        "prix": last_hist_pt["prix"],
        "borne_basse": last_hist_pt["prix"],
        "borne_haute": last_hist_pt["prix"],
        "serie": "Prévision et intervalle"
    }]
    for p in preds_table:
        forecast_points.append({
            "date": p["date"],
            "prix": p["prix_prevu"],
            "borne_basse": p["borne_basse"],
            "borne_haute": p["borne_haute"],
            "serie": "Prévision et intervalle"
        })
    forecast_df = pd.DataFrame(forecast_points)

    hist_line = alt.Chart(hist_chart_df).mark_line(
        color="#2D6A4F", strokeWidth=2.2
    ).encode(
        x=alt.X("date:T", title=None, axis=alt.Axis(format="%Y", labelAngle=0, tickCount=5)),
        y=alt.Y("prix:Q", title=None, scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("date:T", format="%B %Y", title="Date"), alt.Tooltip("prix:Q", format=".0f", title="Prix observé (FCFA)")]
    )

    pred_line = alt.Chart(forecast_df).mark_line(
        color="#D97706", strokeDash=[5, 5], strokeWidth=2.2
    ).encode(
        x=alt.X("date:T"),
        y=alt.Y("prix:Q"),
        tooltip=[alt.Tooltip("date:T", format="%B %Y", title="Date prévision"), alt.Tooltip("prix:Q", format=".0f", title="Prix prévu (FCFA)")]
    )

    band = alt.Chart(forecast_df).mark_area(
        color="#D97706", opacity=0.18
    ).encode(
        x=alt.X("date:T"),
        y=alt.Y("borne_basse:Q"),
        y2=alt.Y2("borne_haute:Q")
    )

    final_chart = (band + hist_line + pred_line).properties(
        height=220
    ).configure_view(
        strokeOpacity=0
    ).configure_axis(
        gridColor="#F3F4F6",
        domainColor="#E5E7EB"
    )

    st.altair_chart(final_chart, width="stretch")

    st.markdown("""
        <div style="display: flex; gap: 20px; font-size: 11px; color: #4B5563; margin-top: 4px;">
            <div><span style="color: #2D6A4F; font-size: 14px; font-weight: bold;">―</span> Historique observé</div>
            <div><span style="color: #D97706; font-size: 14px; font-weight: bold;">┄</span> Prévision et intervalle</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- DEUX TABLEAUX INFÉRIEURS ---
    col_tab_left, col_tab_right = st.columns(2)

    with col_tab_left:
        st.markdown("""
        <div class="section-box" style="margin-bottom: 0px;">
            <div class="section-title">Prévisions mois par mois</div>
        """, unsafe_allow_html=True)
        
        df_mois = pd.DataFrame([
            {
                "Mois": p["mois_nom"],
                "Prix prévu": f"{int(round(p['prix_prevu']))} FCFA",
                "Intervalle": p["intervalle_str"]
            }
            for p in preds_table
        ]).astype(str)
        st.dataframe(df_mois, width="stretch", hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_tab_right:
        st.markdown(f"""
        <div class="section-box" style="margin-bottom: 0px;">
            <div class="section-title">Comparaison des marchés ({selected_commo_display.lower()})</div>
        """, unsafe_allow_html=True)

        df_comp = compare_markets_h1(
            commodity=selected_commodity_raw,
            main_market=selected_market_raw,
            other_markets=other_markets_to_compare,
            model_choice=model_choice,
            df_prices=df_prices,
            horizon=horizon_num,
            models_dir=models_dir
        ).astype(str)
        st.dataframe(df_comp, width="stretch", hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # --- PIED DE PAGE DISCRET ---
    active_model_name = metadata.get("modele_final", "random_forest")
    v_str = metadata.get("version", "1.0.0")
    date_tr = metadata.get("date_entrainement", "2026-11-06")
    mae_test = metadata.get("metriques", {}).get(active_model_name, {}).get("h1", {}).get("MAE", 55.09)

    st.markdown(f"""
    <div class="app-footer">
        Modèle : <strong>{active_model_name}</strong> · version <strong>{v_str}</strong> · entraîné le <strong>{date_tr}</strong> · erreur moyenne sur le test : <strong>{mae_test} FCFA/kg</strong>.
    </div>
    """, unsafe_allow_html=True)

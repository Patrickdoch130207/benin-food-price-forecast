"""Script d'entraînement et d'évaluation des modèles statistiques de référence (Partie 5).

Usage :
    python train_baselines.py
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np

from src.baselines import (
    train_and_export_sarima_models,
    evaluate_all_baselines,
    KEY_COMMODITIES,
    KEY_MARKETS
)
from src.inference import NOMS_PRODUITS


def main():
    print("=" * 75)
    print("ENTRAÎNEMENT & ÉVALUATION DES MODÈLES DE RÉFÉRENCE (BASELINES)")
    print("  1. Naïf (prix du dernier mois)")
    print("  2. Naïf Saisonnier (prix du même mois l'an passé)")
    print("  3. SARIMA (un modèle par série produit x marché sur échantillon représentatif)")
    print("=" * 75)

    data_path = Path("data/prix_propres.parquet")
    if not data_path.exists():
        data_path = Path("benin-food-price-forecast/data/prix_propres.parquet")

    df_prices = pd.read_parquet(data_path)
    models_dir = Path("models") if Path("models").exists() else Path("benin-food-price-forecast/models")

    # 1. Entraînement et export des modèles SARIMA
    print("\n[PHASE 1] Entraînement des modèles SARIMA par série...")
    artifacts = train_and_export_sarima_models(
        df_prices=df_prices,
        test_cutoff="2025-01-01",
        commodities=KEY_COMMODITIES,
        markets=KEY_MARKETS,
        output_dir=str(models_dir)
    )

    # 2. Évaluation comparative sur le jeu de test scellé (2025–2026)
    print("\n[PHASE 2] Évaluation sur le jeu de test scellé (2025–2026)...")
    eval_scores = evaluate_all_baselines(
        df_prices=df_prices,
        test_cutoff="2025-01-01",
        commodities=KEY_COMMODITIES,
        markets=KEY_MARKETS
    )

    summary_rows = []
    for comm in KEY_COMMODITIES:
        nom_fr = NOMS_PRODUITS.get(comm, comm)
        sc = eval_scores.get(comm, {})
        summary_rows.append({
            "Produit": nom_fr,
            "Naïf (MAE)": sc.get("MAE_Naif", np.nan),
            "Naïf Saisonnier (MAE)": sc.get("MAE_Naif_Saisonnier", np.nan),
            "SARIMA (MAE)": sc.get("MAE_SARIMA", np.nan),
        })

    summary_df = pd.DataFrame(summary_rows)
    print("\nTableau comparatif des Baselines sur le Test (2025–2026) :")
    print(summary_df.to_string(index=False))

    # 3. Mise à jour de metadata.json avec les résultats SARIMA et Naïf Saisonnier
    meta_path = models_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # Calcul des moyennes globales
        mae_sarima_mean = round(float(summary_df["SARIMA (MAE)"].dropna().mean()), 2)
        mae_sais_mean = round(float(summary_df["Naïf Saisonnier (MAE)"].dropna().mean()), 2)

        meta.setdefault("metriques", {})
        meta["metriques"]["sarima"] = {
            "h1": {"MAE": mae_sarima_mean},
            "par_produit": {row["Produit"]: row["SARIMA (MAE)"] for _, row in summary_df.iterrows()}
        }
        meta["metriques"]["naif_saisonnier"] = {
            "h1": {"MAE": mae_sais_mean},
            "par_produit": {row["Produit"]: row["Naïf Saisonnier (MAE)"] for _, row in summary_df.iterrows()}
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"\n[PHASE 3] metadata.json mis à jour avec les métriques SARIMA et Naïf Saisonnier.")

    print("\n" + "=" * 75)
    print("ENTRAÎNEMENT DES BASELINES TERMINÉ AVEC SUCCÈS !")
    print("=" * 75)


if __name__ == "__main__":
    main()


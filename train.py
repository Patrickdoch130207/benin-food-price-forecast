"""Script d'entraînement et d'évaluation complet des modèles ML sur le Feature Engineering unifié.

Usage :
    python train.py
"""

from pathlib import Path
import os
import warnings
import pandas as pd
import numpy as np

from src.features import prepare_food_price_features, FEATURES, NOMS_PRODUITS
from src.split import get_train_test_split, rolling_origin_split
from src.metrics import (
    compute_metrics,
    evaluate_baseline_naive,
    evaluate_baseline_seasonal_naive,
    get_naive_predictions,
    get_seasonal_naive_predictions,
)
from src.models import (
    get_default_feature_names,
    optimize_model_rolling_origin,
    train_model_for_horizon,
    export_model_artifact,
    save_params_json,
    generate_metadata,
    LIGHTGBM_AVAILABLE,
)
from src.predict import predict_next_price

warnings.filterwarnings("ignore", category=UserWarning)


def main():
    print("=" * 75, flush=True)
    print("PRÉVISION DES PRIX ALIMENTAIRES AU BÉNIN — PIPELINE UNIFIÉ (EDA + FEATURES)", flush=True)
    print("=" * 75, flush=True)

    # 1. Construction du jeu de données de features unifié (03_features.ipynb / src/features.py)
    print("\n1. Génération du jeu de données de features unifié (src/features.py)...", flush=True)
    df_ml = prepare_food_price_features()

    df_ml["date"] = pd.to_datetime(df_ml["date"])
    print(f"   Dimensions : {df_ml.shape[0]:,} lignes, {df_ml.shape[1]} colonnes", flush=True)
    print(f"   Séries     : {df_ml.groupby(['commodity', 'market']).ngroups} séries sur {df_ml['market'].nunique()} marchés", flush=True)
    print(f"   Période    : {df_ml['date'].min().date()} à {df_ml['date'].max().date()}", flush=True)

    # 2. Découpage temporel strict (Train 2020-2024, Test scellé 2025-2026)
    train_all_df, test_df = get_train_test_split(df_ml, test_cutoff="2025-01-01")
    train_df = train_all_df[train_all_df["date"] >= "2020-01-01"].copy()
    print(f"\n2. Découpage temporel sans fuite :", flush=True)
    print(f"   Train (2020–2024) : {train_df['date'].min().date()} à {train_df['date'].max().date()} ({len(train_df):,} lignes)", flush=True)
    print(f"   Test scellé       : {test_df['date'].min().date()} à {test_df['date'].max().date()} ({len(test_df):,} lignes)", flush=True)

    # 3. Plis Rolling Origin sur la période d'entraînement
    plis = rolling_origin_split(train_df, test_cutoff="2025-01-01", min_train_months=36, val_months=3, step_months=6)
    print(f"\n3. Validation temporelle : {len(plis)} plis Rolling Origin (Walk-Forward) générés.", flush=True)

    # 4. Variables explicatives (2 catégorielles + 10 numériques issues de 02_eda & 03_features)
    cat_cols, num_cols = get_default_feature_names(train_df)
    features = cat_cols + num_cols
    print(f"\n4. Variables explicatives ({len(features)} au total) :", flush=True)
    print(f"   Catégorielles ({len(cat_cols)}) : {cat_cols}", flush=True)
    print(f"   Numériques    ({len(num_cols)}) : {num_cols}", flush=True)

    # 5. Grille d'optimisation d'hyperparamètres
    rf_configs = [
        {"criterion": "absolute_error", "n_estimators": 150, "max_depth": 7, "min_samples_leaf": 12, "max_features": 0.7, "shrinkage": 0.65},
        {"criterion": "absolute_error", "n_estimators": 150, "max_depth": 8, "min_samples_leaf": 8, "max_features": 0.7, "shrinkage": 0.60},
    ]

    lgbm_configs = [
        {"objective": "mae", "n_estimators": 120, "learning_rate": 0.03, "max_depth": 4, "num_leaves": 12, "min_child_samples": 30, "subsample": 0.8, "colsample_bytree": 0.8, "shrinkage": 0.85},
        {"objective": "mae", "n_estimators": 150, "learning_rate": 0.03, "max_depth": 5, "num_leaves": 15, "min_child_samples": 25, "subsample": 0.8, "colsample_bytree": 0.8, "shrinkage": 0.75},
    ] if LIGHTGBM_AVAILABLE else []

    horizons = [1, 2, 3]
    algos = ["random_forest"] + (["lightgbm"] if LIGHTGBM_AVAILABLE else [])

    best_configs = {}
    validation_summary = []

    print("\n" + "-" * 75, flush=True)
    print("PHASE 1 : OPTIMISATION ET VALIDATION ROLLING ORIGIN (2020–2024)", flush=True)
    print("-" * 75, flush=True)

    for h in horizons:
        base_scores = evaluate_baseline_naive(train_df.dropna(subset=[f"target_{h}m"]), horizon=h)
        validation_summary.append({
            "Horizon": f"H{h}",
            "Modèle": "Naïf (prix_m1)",
            **base_scores
        })

        for algo in algos:
            configs = rf_configs if algo == "random_forest" else lgbm_configs
            best_cfg, results_df = optimize_model_rolling_origin(
                train_df, algo, h, configs, plis, cat_cols, num_cols
            )
            best_configs[(algo, h)] = best_cfg
            best_mae = results_df.iloc[0]["MAE_moyenne"]
            validation_summary.append({
                "Horizon": f"H{h}",
                "Modèle": algo,
                "MAE": best_mae,
                "RMSE": np.nan,
                "MAPE (%)": np.nan,
                "WAPE (%)": np.nan
            })
            print(f"  Horizon H{h} | {algo:14s} | MAE validation Rolling Origin = {best_mae:.2f} FCFA", flush=True)

    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    params_file = save_params_json(best_configs, model_dir=model_dir)
    print(f"\nHyperparamètres optimaux enregistrés dans : {params_file}", flush=True)

    # 6. Entraînement final et évaluation sur le Test scellé (2025–2026)
    print("\n" + "-" * 75, flush=True)
    print("PHASE 2 : ÉVALUATION SUR LE TEST FINAL SCELLÉ (2025–2026)", flush=True)
    print("-" * 75, flush=True)

    exported_files = []
    test_metrics_dict = {}
    selected_family_by_h = {}
    test_summary = []
    baseline_metrics = {"naif": {}, "naif_saisonnier": {}}
    metrics_by_product = {"naif": {}, "naif_saisonnier": {}, "random_forest": {}, "lightgbm": {}}

    for h in horizons:
        target_col = f"target_{h}m"
        test_h = test_df.dropna(subset=[target_col]).copy()

        # Baselines sur le test
        base_naive = evaluate_baseline_naive(test_h, horizon=h)
        base_snaive = evaluate_baseline_seasonal_naive(test_h, horizon=h)
        baseline_metrics["naif"][f"h{h}"] = base_naive
        baseline_metrics["naif_saisonnier"][f"h{h}"] = base_snaive

        test_summary.append({"Horizon": f"H{h}", "Modèle": "Naïf (M-1)", **base_naive})
        test_summary.append({"Horizon": f"H{h}", "Modèle": "Naïf saisonnier (M-12)", **base_snaive})

        if h == 1:
            test_h["pred_naif"] = get_naive_predictions(test_h)
            test_h["pred_saisonnier"] = get_seasonal_naive_predictions(test_h)
            for prod_en, prod_fr in NOMS_PRODUITS.items():
                sub_p = test_h[test_h["commodity"] == prod_en]
                if not sub_p.empty:
                    metrics_by_product["naif"][prod_fr] = compute_metrics(sub_p[target_col], sub_p["pred_naif"])["MAE"]
                    metrics_by_product["naif_saisonnier"][prod_fr] = compute_metrics(sub_p[target_col], sub_p["pred_saisonnier"])["MAE"]

        h_candidates = []
        for algo in algos:
            cfg = best_configs.get((algo, h))
            pipe, feat_list = train_model_for_horizon(train_df, algo, h, cfg, cat_cols, num_cols)
            pred = pipe.predict(test_h[features])
            scores = compute_metrics(test_h[target_col], pred)

            art_path = export_model_artifact(pipe, features, algo, h, model_dir=model_dir)
            exported_files.append(os.path.basename(art_path))

            key = f"{algo}_h{h}"
            test_metrics_dict[key] = scores

            test_summary.append({"Horizon": f"H{h}", "Modèle": algo, **scores})
            h_candidates.append((algo, scores["MAE"]))

            if h == 1:
                test_h[f"pred_{algo}"] = pred
                for prod_en, prod_fr in NOMS_PRODUITS.items():
                    sub_p = test_h[test_h["commodity"] == prod_en]
                    if not sub_p.empty:
                        metrics_by_product[algo][prod_fr] = compute_metrics(sub_p[target_col], sub_p[f"pred_{algo}"])["MAE"]

        best_algo_h = min(h_candidates, key=lambda x: x[1])[0]
        selected_family_by_h[h] = best_algo_h

    test_results_df = pd.DataFrame(test_summary)
    print("\nScores globaux sur l'ensemble de Test scellé (2025–2026) :", flush=True)
    print(test_results_df.to_string(index=False), flush=True)

    print("\nComparaison MAE par produit à Horizon 1 mois (FCFA/kg ou L) :", flush=True)
    prod_df = pd.DataFrame(metrics_by_product)
    print(prod_df.to_string(), flush=True)

    # 7. Génération du metadata.json
    meta_path = generate_metadata(
        model_dir=model_dir,
        train_df=train_df,
        test_df=test_df,
        selected_models=selected_family_by_h,
        metrics_by_model=test_metrics_dict,
        exported_files=exported_files,
        baseline_metrics=baseline_metrics,
        metrics_by_product=metrics_by_product,
    )
    print(f"\nMétadonnées mises à jour : {meta_path}", flush=True)

    # 8. Test d'inférence directe
    sample_row = test_df.iloc[0]
    sample_model = os.path.join(model_dir, "random_forest_h1.joblib")
    pred_val = predict_next_price(sample_row, sample_model)
    real_val = sample_row["target_1m"]
    print(f"\nTest d'inférence directe ({sample_row['produit']} à {sample_row['market']}, {sample_row['date'].date()}) :", flush=True)
    print(f"  Prédit H1 = {pred_val:.1f} FCFA | Réel = {real_val:.1f} FCFA | Écart = {abs(pred_val - real_val):.1f} FCFA", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()

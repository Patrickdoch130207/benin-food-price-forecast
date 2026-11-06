"""Script d'entraînement et d'évaluation complet des modèles ML.

Usage :
    python train.py
"""

from pathlib import Path
import os
import pandas as pd
import numpy as np

from src.features import prepare_food_price_features
from src.split import get_train_test_split, rolling_origin_split
from src.metrics import compute_metrics, compute_metrics_by_group, evaluate_baseline_naive
from src.models import (
    get_default_feature_names,
    optimize_model_rolling_origin,
    train_model_for_horizon,
    export_model_artifact,
    generate_metadata,
    LIGHTGBM_AVAILABLE
)
from src.predict import predict_next_price


def main():
    print("=" * 70, flush=True)
    print("PRÉVISION DES PRIX ALIMENTAIRES AU BÉNIN — ENTRAÎNEMENT DES MODÈLES", flush=True)
    print("=" * 70, flush=True)

    # 1. Régénération systématique des features (intègre les dernières modifications de Patrick)
    print("\nGénération du jeu de données de features...", flush=True)
    df_ml = prepare_food_price_features()

    df_ml["date"] = pd.to_datetime(df_ml["date"])
    print(f"Dimensions : {df_ml.shape[0]} lignes, {df_ml.shape[1]} colonnes", flush=True)
    print(f"Période    : {df_ml['date'].min().date()} à {df_ml['date'].max().date()}", flush=True)

    # 2. Découpage temporel strict (Train < 2025, Test final >= 2025)
    train_df, test_df = get_train_test_split(df_ml, test_cutoff="2025-01-01")
    print(f"\n[DÉCOUPAGE TEMPOREL]", flush=True)
    print(f"  Train (Dev) : {train_df['date'].min().date()} à {train_df['date'].max().date()} ({len(train_df):,} lignes)", flush=True)
    print(f"  Test final  : {test_df['date'].min().date()} à {test_df['date'].max().date()} ({len(test_df):,} lignes)", flush=True)

    # 3. Plis Rolling Origin sur la période d'entraînement (2020-2024)
    plis = rolling_origin_split(train_df, test_cutoff="2025-01-01", min_train_months=48, val_months=3, step_months=3)
    print(f"\n[VALIDATION TEMPORELLE] {len(plis)} plis Rolling Origin (Walk-Forward) générés.", flush=True)

    # 4. Identification des variables
    cat_cols, num_cols = get_default_feature_names(train_df)
    features = cat_cols + num_cols
    print(f"\n[VARIABLES] {len(features)} au total ({len(cat_cols)} catégorielles : {cat_cols}, {len(num_cols)} numériques)", flush=True)

    # 5. Configurations d'hyperparamètres
    rf_configs = [
        {"n_estimators": 100, "max_depth": 15, "min_samples_leaf": 2, "max_features": "sqrt"}
    ]

    lgbm_configs = [
        {"n_estimators": 250, "learning_rate": 0.05, "max_depth": 6, "num_leaves": 31}
    ] if LIGHTGBM_AVAILABLE else []

    horizons = [1, 2, 3]
    algos = ["random_forest"] + (["lightgbm"] if LIGHTGBM_AVAILABLE else [])

    best_configs = {}
    validation_summary = []

    print("\n" + "-" * 70, flush=True)
    print("PHASE 1 : OPTIMISATION ET VALIDATION ROLLING ORIGIN", flush=True)
    print("-" * 70, flush=True)

    for h in horizons:
        # Évaluation de la baseline naïve pour comparaison
        base_scores = evaluate_baseline_naive(train_df, horizon=h)
        validation_summary.append({
            "Horizon": f"H{h}",
            "Modèle": "Naïf (lag_1)",
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

    print("\nSynthèse de la validation :", flush=True)
    print(pd.DataFrame(validation_summary).to_string(index=False), flush=True)

    # 6. Entraînement final et évaluation sur le Test scellé (2025–2026)
    print("\n" + "-" * 70, flush=True)
    print("PHASE 2 : ÉVALUATION SUR LE TEST FINAL SCELLÉ (2025–2026)", flush=True)
    print("-" * 70, flush=True)

    exported_files = []
    test_metrics_dict = {}
    selected_family_by_h = {}
    test_summary = []
    baseline_metrics = {"naif": {}}

    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    for h in horizons:
        target_col = f"target_{h}m"
        test_h = test_df.dropna(subset=[target_col]).copy()

        # Score naïf sur test
        base_test_scores = evaluate_baseline_naive(test_h, horizon=h)
        test_summary.append({
            "Horizon": f"H{h}",
            "Modèle": "Naïf (lag_1)",
            **base_test_scores
        })
        baseline_metrics["naif"][f"h{h}"] = base_test_scores

        h_candidates = []
        for algo in algos:
            cfg = best_configs.get((algo, h))
            pipe, feat_list = train_model_for_horizon(train_df, algo, h, cfg, cat_cols, num_cols)
            pred = pipe.predict(test_h[features])
            scores = compute_metrics(test_h[target_col], pred)

            # Export de l'artefact pour Donald
            art_path = export_model_artifact(pipe, features, algo, h, model_dir=model_dir)
            exported_files.append(os.path.basename(art_path))

            key = f"{algo}_h{h}"
            test_metrics_dict[key] = scores

            test_summary.append({
                "Horizon": f"H{h}",
                "Modèle": algo,
                **scores
            })
            h_candidates.append((algo, scores["MAE"]))

        # Meilleur modèle sur cet horizon
        best_algo_h = min(h_candidates, key=lambda x: x[1])[0]
        selected_family_by_h[h] = best_algo_h

    test_results_df = pd.DataFrame(test_summary)
    print("\nScores sur l'ensemble de Test final (2025-2026) :", flush=True)
    print(test_results_df.to_string(index=False), flush=True)

    # 7. Génération du metadata.json
    meta_path = generate_metadata(
        model_dir=model_dir,
        train_df=train_df,
        test_df=test_df,
        selected_models=selected_family_by_h,
        metrics_by_model=test_metrics_dict,
        exported_files=exported_files,
        baseline_metrics=baseline_metrics
    )
    print(f"\nMétadonnées générées : {meta_path}", flush=True)

    # 8. Test de la fonction d'inférence (Donald)
    print("\n" + "-" * 70, flush=True)
    print("PHASE 3 : TEST D'INFÉRENCE DIRECTE", flush=True)
    print("-" * 70, flush=True)
    sample_row = test_df.iloc[0]
    sample_model = os.path.join(model_dir, "random_forest_h1.joblib")
    pred_val = predict_next_price(sample_row, sample_model)
    real_val = sample_row["target_1m"]

    print(f"Produit        : {sample_row['commodity']}", flush=True)
    print(f"Marché         : {sample_row['market']}", flush=True)
    print(f"Date           : {sample_row['date'].date()}", flush=True)
    print(f"Prix prédit H1 : {pred_val:.2f} FCFA", flush=True)
    print(f"Prix réel H1   : {real_val:.2f} FCFA", flush=True)
    print(f"Écart absolu   : {abs(pred_val - real_val):.2f} FCFA", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("PIPELINE D'ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS !", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()

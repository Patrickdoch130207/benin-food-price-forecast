"""Script principal de réentraînement automatique mensuel des modèles prédictifs.

Usage :
    python scripts/retrain.py [--source auto|local|url] [--holdout-months 6] [--dry-run] [--force]
"""

from pathlib import Path
import argparse
import datetime
import logging
import sys
import pandas as pd

# Ajout du dossier racine au PATH
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data_download import fetch_latest_food_prices, load_config
from src.data_validation import validate_raw_food_prices
from src.features import prepare_data, prepare_food_price_features, NOMS_PRODUITS
from src.metrics import compute_metrics
from src.models import train_all
from src.model_registry import (
    read_metadata,
    write_metadata,
    bump_version,
    archive_current_models,
    promote_candidate_models,
    append_retrain_log,
    evaluate_decision_rule,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s : %(message)s"
)
logger = logging.getLogger("retrain_pipeline")


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline de réentraînement automatique mensuel.")
    parser.add_argument("--source", choices=["auto", "local", "url"], default="auto", help="Source des données.")
    parser.add_argument("--holdout-months", type=int, default=6, help="Nombre de mois récents réservés pour l'évaluation.")
    parser.add_argument("--dry-run", action="store_true", help="Exécute l'évaluation sans remplacer les modèles en service.")
    parser.add_argument("--force", action="store_true", help="Force le remplacement du modèle même si le candidat n'est pas meilleur.")
    parser.add_argument("--config", type=str, default="config/retrain.yaml", help="Chemin du fichier de configuration.")
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("=" * 70)
    logger.info("DÉMARRAGE DU WORKFLOW DE RÉENTRAÎNEMENT AUTOMATIQUE")
    logger.info("=" * 70)

    config = load_config(args.config)
    paths_cfg = config.get("paths", {})
    models_dir = paths_cfg.get("models_dir", "models")
    candidate_dir = paths_cfg.get("candidate_dir", "models_candidate")
    archive_dir = paths_cfg.get("archive_dir", "models/archive")
    log_path = paths_cfg.get("log_file", "models/retrain_log.jsonl")

    current_meta = read_metadata(models_dir)
    current_version = current_meta.get("version", "1.0.0")
    current_max_date = current_meta.get("date_fin_test")

    start_time = datetime.datetime.now()

    # -------------------------------------------------------------------------
    # Étape a : Téléchargement des dernières données
    # -------------------------------------------------------------------------
    logger.info("\n[1/7] Téléchargement des données de prix PAM / HDX...")
    force_local = (args.source == "local")
    try:
        df_raw, saved_raw_path, source_used = fetch_latest_food_prices(
            source_type=args.source,
            config=config,
            force_local=force_local
        )
    except Exception as e:
        logger.error(f"ÉCHEC TECHNIQUE lors du téléchargement : {e}")
        append_retrain_log({
            "timestamp": start_time.isoformat(),
            "action": "retrain",
            "version_before": current_version,
            "decision": "échec",
            "reason": f"Erreur de téléchargement : {e}"
        }, log_path=log_path)
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Étape b : Validation de la qualité des données
    # -------------------------------------------------------------------------
    logger.info("\n[2/7] Validation de la qualité et du schéma des données...")
    val_report = validate_raw_food_prices(
        df_raw,
        min_rows=config.get("validation", {}).get("min_rows", 10000),
        required_columns=config.get("validation", {}).get("required_columns"),
        current_max_date=current_max_date,
        critical_commodities=config.get("validation", {}).get("critical_commodities")
    )

    if not val_report["is_valid"]:
        logger.error("ÉCHEC DE VALIDATION DES DONNÉES :")
        for err in val_report["errors"]:
            logger.error(f"  - {err}")
        append_retrain_log({
            "timestamp": start_time.isoformat(),
            "action": "retrain",
            "version_before": current_version,
            "data_rows": len(df_raw),
            "data_source": source_used,
            "decision": "échec",
            "reason": f"Validation rejetée : {'; '.join(val_report['errors'])}"
        }, log_path=log_path)
        logger.warning("Arrêt propre : AUCUNE modification n'a été portée sur les modèles en service.")
        sys.exit(2)

    for w in val_report["warnings"]:
        logger.warning(f"  [Avertissement] {w}")

    # -------------------------------------------------------------------------
    # Étape c : Prétraitement unifié (prepare_data)
    # -------------------------------------------------------------------------
    logger.info("\n[3/7] Prétraitement et nettoyage unifié des données...")
    df_clean = prepare_data(df_raw)
    clean_path = paths_cfg.get("processed_data", "data/prix_propres.parquet")
    Path(clean_path).parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(clean_path, index=False)
    logger.info(f"Dataset propre généré : {len(df_clean):,} lignes (sauvegardé dans {clean_path}).")

    # -------------------------------------------------------------------------
    # Étape d : Construction des features et isolation du holdout
    # -------------------------------------------------------------------------
    logger.info("\n[4/7] Construction des features unifiées sans fuite...")
    features_path = paths_cfg.get("features_data", "data/features.parquet")
    df_features = prepare_food_price_features(df_propre=df_clean, features_path=features_path)

    # Découpage du holdout (N derniers mois)
    holdout_months = args.holdout_months
    df_features["date"] = pd.to_datetime(df_features["date"])
    max_date = df_features["date"].max()
    holdout_cutoff = max_date - pd.DateOffset(months=holdout_months)

    df_train_cand = df_features[df_features["date"] < holdout_cutoff].copy()
    df_holdout = df_features[df_features["date"] >= holdout_cutoff].copy()

    logger.info(f"Découpage Holdout ({holdout_months} mois) :")
    logger.info(f"  Apprentissage candidat : {df_train_cand['date'].min().date()} à {df_train_cand['date'].max().date()} ({len(df_train_cand):,} l.)")
    logger.info(f"  Fenêtre Holdout        : {df_holdout['date'].min().date()} à {df_holdout['date'].max().date()} ({len(df_holdout):,} l.)")

    # Entraînement des modèles candidats dans models_candidate/ (JAMAIS dans models/)
    Path(candidate_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"\n[5/7] Entraînement du candidat dans le dossier isolé '{candidate_dir}/'...")
    train_all(df_train_cand, output_dir=candidate_dir, params_dir=models_dir)

    # -------------------------------------------------------------------------
    # Étape e : Évaluation comparative sur exactement la même fenêtre de holdout
    # -------------------------------------------------------------------------
    logger.info("\n[6/7] Évaluation comparative sur la fenêtre de holdout scellée...")
    import joblib

    # Évaluation du modèle en service (Current)
    curr_rf_art = joblib.load(Path(models_dir) / "random_forest_h1.joblib")
    curr_rf_pipe = curr_rf_art["pipeline"]
    curr_rf_feats = curr_rf_art["features"]

    # Évaluation du candidat (Candidate)
    cand_rf_art = joblib.load(Path(candidate_dir) / "random_forest_h1.joblib")
    cand_rf_pipe = cand_rf_art["pipeline"]
    cand_rf_feats = cand_rf_art["features"]

    holdout_clean = df_holdout.dropna(subset=["target_1m"]).copy()
    y_true = holdout_clean["target_1m"]

    curr_preds = curr_rf_pipe.predict(holdout_clean[curr_rf_feats])
    cand_preds = cand_rf_pipe.predict(holdout_clean[cand_rf_feats])

    curr_scores = compute_metrics(y_true, curr_preds)
    cand_scores = compute_metrics(y_true, cand_preds)

    logger.info(f"  Modèle en service (v{current_version}) : MAE = {curr_scores['MAE']:.2f} F | RMSE = {curr_scores['RMSE']:.2f} F | MAPE = {curr_scores['MAPE (%)']:.2f} %")
    logger.info(f"  Modèle candidat                     : MAE = {cand_scores['MAE']:.2f} F | RMSE = {cand_scores['RMSE']:.2f} F | MAPE = {cand_scores['MAPE (%)']:.2f} %")

    # Évaluation par produit
    curr_prod_maes = {}
    cand_prod_maes = {}
    holdout_clean["curr_p"] = curr_preds
    holdout_clean["cand_p"] = cand_preds
    for prod_en, prod_fr in NOMS_PRODUITS.items():
        sub_p = holdout_clean[holdout_clean["commodity"] == prod_en]
        if not sub_p.empty:
            curr_prod_maes[prod_fr] = compute_metrics(sub_p["target_1m"], sub_p["curr_p"])["MAE"]
            cand_prod_maes[prod_fr] = compute_metrics(sub_p["target_1m"], sub_p["cand_p"])["MAE"]

    # -------------------------------------------------------------------------
    # Étape f : Règle de décision Champion / Challenger
    # -------------------------------------------------------------------------
    rule_cfg = config.get("retraining", {}).get("decision_rule", {})
    should_promote, reason = evaluate_decision_rule(
        curr_scores,
        cand_scores,
        current_prod_metrics=curr_prod_maes,
        candidate_prod_metrics=cand_prod_maes,
        max_mae_ratio=float(rule_cfg.get("max_global_mae_ratio", 1.0)),
        max_prod_degrad_pct=float(rule_cfg.get("max_product_degradation_pct", 15.0))
    )

    if args.force:
        logger.warning("Option --force détectée : forçage du remplacement pour la démonstration.")
        should_promote = True
        reason = f"Remplacement forcé par argument CLI (--force). ({reason})"

    if args.dry_run:
        logger.info("Option --dry-run détectée : simulation terminée sans modification du modèle en service.")
        should_promote = False
        reason = f"Simulation (--dry-run) : {reason}"

    # -------------------------------------------------------------------------
    # Étape g : Promotion, Archivage et Versionnage
    # -------------------------------------------------------------------------
    new_version = current_version
    decision = "conservé"

    if should_promote:
        decision = "remplacé"
        new_version = bump_version(current_version, bump_type="patch")
        logger.info(f"\n[7/7] PROMOTION DU CANDIDAT vers la version v{new_version}...")

        # 1. Archiver l'ancien modèle
        archive_current_models(models_dir=models_dir, archive_dir=archive_dir)

        # 2. Copier le nouveau modèle
        promoted = promote_candidate_models(candidate_dir=candidate_dir, models_dir=models_dir, new_version=new_version)

        # 3. Mettre à jour metadata.json
        updated_meta = current_meta.copy()
        updated_meta["version"] = new_version
        updated_meta["date_entrainement"] = datetime.date.today().strftime("%Y-%m-%d")
        updated_meta["donnees_jusqu_a"] = str(max_date.date())[:7]
        updated_meta["date_fin_train"] = str(df_train_cand["date"].max().date())
        updated_meta["date_debut_test"] = str(df_holdout["date"].min().date())
        updated_meta["date_fin_test"] = str(df_holdout["date"].max().date())
        updated_meta["metriques_holdout_candidat"] = {
            "random_forest": {"h1": cand_scores},
            "par_produit": cand_prod_maes
        }
        write_metadata(updated_meta, models_dir=models_dir)
        logger.info(f"Version mise à jour avec succès : v{new_version}")
    else:
        logger.info(f"\n[7/7] DÉCISION : Modèle en service conservé (v{current_version}). Raison : {reason}")

    # -------------------------------------------------------------------------
    # Étape h : Journalisation (retrain_log.jsonl)
    # -------------------------------------------------------------------------
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "action": "retrain",
        "data_source": source_used,
        "data_rows": len(df_raw),
        "data_max_date": str(max_date.date()),
        "holdout_months": holdout_months,
        "version_before": current_version,
        "version_after": new_version,
        "current_mae": curr_scores["MAE"],
        "candidate_mae": cand_scores["MAE"],
        "decision": decision,
        "reason": reason,
        "dry_run": args.dry_run,
        "forced": args.force
    }
    append_retrain_log(log_entry, log_path=log_path)
    logger.info(f"Événement consigné dans le journal : {log_path}")

    logger.info("=" * 70)
    logger.info(f"PIPELINE DE RÉENTRAÎNEMENT TERMINÉ : Décision = {decision.upper()}")
    logger.info("=" * 70)
    sys.exit(0)


if __name__ == "__main__":
    main()

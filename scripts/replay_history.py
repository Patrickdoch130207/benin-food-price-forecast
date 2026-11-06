"""Simulation de rejeu de l'historique (Backtest séquentiel mensuel).

Prouve la capacité du système à réentraîner, évaluer et décider mois par mois
sans attendre l'arrivée de nouvelles données réelles en production.
Usage :
    python scripts/replay_history.py [--start-date 2024-06-01] [--steps 6] [--output-dir models_simulation]
"""

from pathlib import Path
import argparse
import datetime
import json
import logging
import sys
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.features import prepare_food_price_features
from src.metrics import compute_metrics
from src.models import train_all
from src.model_registry import (
    read_metadata,
    write_metadata,
    bump_version,
    append_retrain_log,
    evaluate_decision_rule,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s : %(message)s"
)
logger = logging.getLogger("replay_history")


def parse_args():
    parser = argparse.ArgumentParser(description="Rejeu séquentiel de l'historique mois par mois.")
    parser.add_argument("--start-date", type=str, default="2024-06-01", help="Date de départ de la simulation (YYYY-MM-DD).")
    parser.add_argument("--steps", type=int, default=5, help="Nombre de mois successifs à rejouer.")
    parser.add_argument("--holdout-months", type=int, default=3, help="Mois de test pour chaque itération.")
    parser.add_argument("--output-dir", type=str, default="models_simulation", help="Dossier isolé pour la simulation.")
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("=" * 70)
    logger.info("DÉBUT DU REJEU HISTORIQUE SÉQUENTIEL (DÉMONSTRATION)")
    logger.info(f"Point de départ : {args.start_date} | Nombre d'itérations : {args.steps}")
    logger.info("=" * 70)

    sim_dir = Path(args.output_dir)
    sim_dir.mkdir(parents=True, exist_ok=True)
    sim_log = sim_dir / "replay_log.jsonl"
    sim_cand = sim_dir / "candidate"
    sim_cand.mkdir(parents=True, exist_ok=True)

    # 1. Chargement des features complètes
    df_features = prepare_food_price_features()
    df_features["date"] = pd.to_datetime(df_features["date"])

    current_date = pd.Timestamp(args.start_date)
    version = "1.0.0"

    # Initialisation du modèle de départ sur données < start_date
    init_train = df_features[df_features["date"] < current_date].copy()
    logger.info(f"Entraînement initial sur {len(init_train):,} lignes (< {current_date.date()})...")
    train_all(init_train, output_dir=str(sim_dir), params_dir="models")

    decisions_summary = []

    for step in range(1, args.steps + 1):
        # On avance d'un mois
        current_date = current_date + pd.DateOffset(months=1)
        logger.info(f"\n--- [ITÉRATION {step}/{args.steps}] Simulation à la date : {current_date.date()} ---")

        available_data = df_features[df_features["date"] <= current_date].copy()
        holdout_cutoff = current_date - pd.DateOffset(months=args.holdout_months)

        train_set = available_data[available_data["date"] < holdout_cutoff].copy()
        holdout_set = available_data[available_data["date"] >= holdout_cutoff].dropna(subset=["target_1m"]).copy()

        if len(holdout_set) == 0:
            logger.warning(f"Pas de données holdout à {current_date.date()}, passage.")
            continue

        # Entraînement du candidat
        train_all(train_set, output_dir=str(sim_cand), params_dir="models")

        # Comparaison
        import joblib
        curr_art = joblib.load(sim_dir / "random_forest_h1.joblib")
        cand_art = joblib.load(sim_cand / "random_forest_h1.joblib")

        curr_preds = curr_art["pipeline"].predict(holdout_set[curr_art["features"]])
        cand_preds = cand_art["pipeline"].predict(holdout_set[cand_art["features"]])

        curr_score = compute_metrics(holdout_set["target_1m"], curr_preds)
        cand_score = compute_metrics(holdout_set["target_1m"], cand_preds)

        promoted, reason = evaluate_decision_rule(curr_score, cand_score, max_mae_ratio=1.0)

        decision_str = "remplacé" if promoted else "conservé"
        if promoted:
            old_v = version
            version = bump_version(version, bump_type="patch")
            for f in sim_cand.glob("*.joblib"):
                import shutil
                shutil.copy2(f, sim_dir / f.name)
            logger.info(f"  >>> CANDIDAT SUPÉRIEUR : Promu v{version} (MAE {cand_score['MAE']:.2f} <= {curr_score['MAE']:.2f})")
        else:
            logger.info(f"  >>> MODÈLE CONSERVÉ : v{version} (MAE {curr_score['MAE']:.2f} vs {cand_score['MAE']:.2f})")

        rec = {
            "step": step,
            "date": str(current_date.date()),
            "current_mae": curr_score["MAE"],
            "candidate_mae": cand_score["MAE"],
            "decision": decision_str,
            "version": version,
            "reason": reason
        }
        decisions_summary.append(rec)
        append_retrain_log(rec, log_path=str(sim_log))

    logger.info("\n" + "=" * 70)
    logger.info("BILAN DU REJEU HISTORIQUE :")
    logger.info("=" * 70)
    df_res = pd.DataFrame(decisions_summary)
    print(df_res.to_string(index=False))


if __name__ == "__main__":
    main()


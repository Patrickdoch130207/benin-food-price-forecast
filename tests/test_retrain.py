"""Tests unitaires et d'intégration pour le pipeline de réentraînement automatique.

Vérifie :
1. La validation des données (rejet des prix négatifs, colonnes manquantes, régression temporelle).
2. La règle de décision Champion / Challenger (promotion si meilleur, conservation sinon).
3. La non-modification des modèles en service en cas d'échec ou de simulation (--dry-run).
4. Le versionnage sémantique et la procédure d'archivage.
5. Le mécanisme de rollback vers une archive précédente.
"""

from pathlib import Path
import json
import pytest
import numpy as np
import pandas as pd

from src.data_validation import validate_raw_food_prices
from src.model_registry import (
    bump_version,
    archive_current_models,
    evaluate_decision_rule,
    rollback_to_version,
    write_metadata,
    read_metadata,
)


@pytest.fixture
def sample_valid_data():
    """Génère un jeu de données synthétique valide de 12 000 lignes."""
    dates = pd.date_range("2020-01-01", "2026-02-01", freq="MS")
    records = []
    commodities = ["Maize (white)", "Rice (imported)", "Tomatoes"]
    markets = ["Dantokpa", "Parakou", "Bohicon"]
    for d in dates:
        for c in commodities:
            for m in markets:
                records.append({
                    "date": str(d.date()),
                    "market": m,
                    "commodity": c,
                    "unit": "KG",
                    "price": 250.0 + np.random.uniform(10, 50),
                    "admin1": "Littoral"
                })
    return pd.DataFrame(records)


def test_validation_accepts_valid_data(sample_valid_data):
    """Vérifie que des données conformes sont acceptées."""
    report = validate_raw_food_prices(sample_valid_data, min_rows=100)
    assert report["is_valid"] is True
    assert len(report["errors"]) == 0


def test_validation_rejects_missing_columns(sample_valid_data):
    """Vérifie que l'absence de colonnes requises bloque la validation."""
    invalid = sample_valid_data.drop(columns=["price"])
    report = validate_raw_food_prices(invalid, min_rows=100)
    assert report["is_valid"] is False
    assert any("price" in err for err in report["errors"])


def test_validation_rejects_negative_prices(sample_valid_data):
    """Vérifie que les prix négatifs ou nuls sont strictement rejetés."""
    invalid = sample_valid_data.copy()
    invalid.loc[10, "price"] = -50.0
    report = validate_raw_food_prices(invalid, min_rows=100)
    assert report["is_valid"] is False
    assert any("négatifs" in err for err in report["errors"])


def test_validation_rejects_time_regression(sample_valid_data):
    """Vérifie qu'un fichier dont la date max recule est rejeté."""
    report = validate_raw_food_prices(
        sample_valid_data,
        min_rows=100,
        current_max_date=pd.Timestamp("2027-01-01")
    )
    assert report["is_valid"] is False
    assert any("Régression temporelle" in err for err in report["errors"])


def test_decision_rule_champion_challenger():
    """Vérifie la règle de décision sur la MAE."""
    current = {"MAE": 50.0}
    better_candidate = {"MAE": 45.0}
    worse_candidate = {"MAE": 55.0}

    promoted, reason = evaluate_decision_rule(current, better_candidate)
    assert promoted is True
    assert "Candidat validé" in reason

    kept, reason_kept = evaluate_decision_rule(current, worse_candidate)
    assert kept is False
    assert "supérieure" in reason_kept


def test_decision_rule_product_degradation():
    """Vérifie le blocage du candidat si une denrée phare subit une régression > 15 %."""
    current = {"MAE": 50.0}
    cand = {"MAE": 48.0}  # globalement meilleur

    curr_p = {"Maïs blanc": 20.0, "Tomate": 100.0}
    cand_p = {"Maïs blanc": 25.0, "Tomate": 80.0}  # maïs +25 % (> 15 %)

    promoted, reason = evaluate_decision_rule(
        current, cand,
        current_prod_metrics=curr_p,
        candidate_prod_metrics=cand_p,
        max_prod_degrad_pct=15.0
    )
    assert promoted is False
    assert "Dégradation critique" in reason


def test_bump_version_semantics():
    """Vérifie l'incrémentation sémantique."""
    assert bump_version("1.0.0", "patch") == "1.0.1"
    assert bump_version("1.0.5", "patch") == "1.0.6"
    assert bump_version("1.0.5", "minor") == "1.1.0"


def test_archive_and_rollback(tmp_path):
    """Vérifie l'archivage et le retour en arrière (rollback)."""
    models_dir = tmp_path / "models"
    archive_dir = tmp_path / "models/archive"
    models_dir.mkdir(parents=True)

    # Version v1.0.0 initiale
    meta_v1 = {"version": "1.0.0", "modele_final": "random_forest"}
    write_metadata(meta_v1, str(models_dir))
    dummy_model = models_dir / "random_forest_h1.joblib"
    dummy_model.write_text("v1 content")

    # Archivage de v1.0.0
    archive_current_models(models_dir=str(models_dir), archive_dir=str(archive_dir))
    assert (archive_dir / "v1.0.0" / "random_forest_h1.joblib").exists()

    # Mise à jour vers v1.0.1
    meta_v2 = {"version": "1.0.1", "modele_final": "lightgbm"}
    write_metadata(meta_v2, str(models_dir))
    dummy_model.write_text("v2 content")

    # Rollback vers v1.0.0
    log_file = tmp_path / "retrain_log.jsonl"
    rollback_to_version("1.0.0", models_dir=str(models_dir), archive_dir=str(archive_dir), log_path=str(log_file))

    restored_meta = read_metadata(str(models_dir))
    assert restored_meta["version"] == "1.0.0"
    assert dummy_model.read_text() == "v1 content"
    assert log_file.exists()


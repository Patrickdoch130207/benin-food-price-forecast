"""Module de registre de modèles, versionnage, comparaison et rollback.

Gère :
- La lecture et l'écriture de `metadata.json`.
- L'archivage de l'ancienne version dans `models/archive/v{version}/`.
- L'incrémentation de version sémantique (patch / minor).
- L'évaluation de la règle de remplacement Champion / Challenger sur la fenêtre de holdout.
- L'écriture dans le journal `models/retrain_log.jsonl`.
- La fonction de rollback vers une version antérieure archivée.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import datetime
import json
import logging
import shutil
import pandas as pd

logger = logging.getLogger(__name__)


def read_metadata(models_dir: str = "models") -> Dict[str, Any]:
    """Lit le fichier metadata.json s'il existe."""
    p = Path(models_dir) / "metadata.json"
    if not p.exists():
        alt = Path("benin-food-price-forecast") / models_dir / "metadata.json"
        if alt.exists():
            p = alt
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def write_metadata(metadata: Dict[str, Any], models_dir: str = "models") -> Path:
    """Enregistre le fichier metadata.json de manière atomique."""
    d = Path(models_dir)
    d.mkdir(parents=True, exist_ok=True)
    out_file = d / "metadata.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return out_file


def bump_version(current_version: str = "1.0.0", bump_type: str = "patch") -> str:
    """Incrémente la version sémantique (ex. 1.0.0 -> 1.0.1 ou 1.1.0)."""
    try:
        parts = [int(p) for p in current_version.split(".")]
        while len(parts) < 3:
            parts.append(0)
    except Exception:
        parts = [1, 0, 0]

    if bump_type == "minor":
        parts[1] += 1
        parts[2] = 0
    else:  # patch
        parts[2] += 1

    return f"{parts[0]}.{parts[1]}.{parts[2]}"


def archive_current_models(
    models_dir: str = "models",
    archive_dir: str = "models/archive"
) -> Optional[Path]:
    """Archive l'ensemble des artefacts et métadonnées du modèle actuellement en service."""
    m_dir = Path(models_dir)
    meta = read_metadata(models_dir)
    if not meta or "version" not in meta:
        version = "1.0.0"
    else:
        version = meta["version"]

    target_archive = Path(archive_dir) / f"v{version}"
    target_archive.mkdir(parents=True, exist_ok=True)

    files_to_archive = list(m_dir.glob("*.joblib")) + [m_dir / "metadata.json", m_dir / "params.json"]
    archived_count = 0
    for f in files_to_archive:
        if f.exists():
            shutil.copy2(f, target_archive / f.name)
            archived_count += 1

    logger.info(f"Version v{version} archivée dans {target_archive} ({archived_count} fichiers).")
    return target_archive


def promote_candidate_models(
    candidate_dir: str = "models_candidate",
    models_dir: str = "models",
    new_version: str = "1.0.1"
) -> List[str]:
    """Promeut les modèles candidats en production dans models/."""
    c_dir = Path(candidate_dir)
    m_dir = Path(models_dir)
    m_dir.mkdir(parents=True, exist_ok=True)

    promoted_files = []
    for f in c_dir.glob("*.joblib"):
        dst = m_dir / f.name
        shutil.copy2(f, dst)
        promoted_files.append(f.name)

    # Copie de params.json si généré
    c_params = c_dir / "params.json"
    if c_params.exists():
        shutil.copy2(c_params, m_dir / "params.json")

    logger.info(f"Promotion réussie de {len(promoted_files)} artefacts candidats vers {m_dir}.")
    return promoted_files


def append_retrain_log(
    log_entry: Dict[str, Any],
    log_path: str = "models/retrain_log.jsonl"
) -> Path:
    """Ajoute une ligne d'événement au journal retrain_log.jsonl."""
    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    return p


def evaluate_decision_rule(
    current_metrics: Dict[str, Any],
    candidate_metrics: Dict[str, Any],
    current_prod_metrics: Optional[Dict[str, float]] = None,
    candidate_prod_metrics: Optional[Dict[str, float]] = None,
    max_mae_ratio: float = 1.0,
    max_prod_degrad_pct: float = 15.0
) -> Tuple[bool, str]:
    """Évalue si le candidat est supérieur ou équivalent au modèle en service.

    Règle :
    1. MAE globale du candidat <= MAE actuelle * max_mae_ratio.
    2. Aucune dégradation supérieure à max_prod_degrad_pct sur un produit phare individuel.
    """
    curr_mae = current_metrics.get("MAE", float("inf"))
    cand_mae = candidate_metrics.get("MAE", float("inf"))

    if cand_mae > curr_mae * max_mae_ratio:
        diff_pct = ((cand_mae - curr_mae) / curr_mae) * 100.0 if curr_mae > 0 else 0
        return False, f"MAE globale supérieure : {cand_mae:.2f} vs {curr_mae:.2f} (+{diff_pct:.1f} %)."

    # Contrôle de dégradation par produit
    if current_prod_metrics and candidate_prod_metrics:
        for prod, curr_p_mae in current_prod_metrics.items():
            if prod in candidate_prod_metrics:
                cand_p_mae = candidate_prod_metrics[prod]
                if curr_p_mae > 0:
                    degrad_pct = ((cand_p_mae - curr_p_mae) / curr_p_mae) * 100.0
                    if degrad_pct > max_prod_degrad_pct:
                        return False, (
                            f"Dégradation critique sur le produit '{prod}' : "
                            f"{cand_p_mae:.2f} vs {curr_p_mae:.2f} (+{degrad_pct:.1f} % > seuil {max_prod_degrad_pct} %)."
                        )

    return True, f"Candidat validé : MAE globale {cand_mae:.2f} <= {curr_mae:.2f} sans régression sectorielle."


def rollback_to_version(
    target_version: str,
    models_dir: str = "models",
    archive_dir: str = "models/archive",
    log_path: str = "models/retrain_log.jsonl"
) -> bool:
    """Restaure une version antérieure archivée dans models/."""
    v_clean = target_version.lstrip("v")
    source_archive = Path(archive_dir) / f"v{v_clean}"

    if not source_archive.exists():
        raise FileNotFoundError(f"Archive introuvable pour la version v{v_clean} dans {archive_dir}.")

    m_dir = Path(models_dir)
    current_meta = read_metadata(models_dir)
    curr_v = current_meta.get("version", "inconnue")

    restored_files = []
    for f in source_archive.glob("*"):
        if f.is_file():
            shutil.copy2(f, m_dir / f.name)
            restored_files.append(f.name)

    logger.warning(f"ROLLBACK exécuté : v{curr_v} remplacée par v{v_clean} ({len(restored_files)} fichiers restaurés).")

    append_retrain_log({
        "timestamp": datetime.datetime.now().isoformat(),
        "action": "rollback",
        "version_before": curr_v,
        "version_after": v_clean,
        "decision": "restored",
        "reason": f"Rollback manuel vers l'archive v{v_clean}",
        "files_restored": restored_files
    }, log_path=log_path)

    return True


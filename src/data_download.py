"""Module de téléchargement automatique des relevés de prix PAM / HDX.

Supporte :
1. Téléchargement direct depuis l'URL de ressource CSV HDX.
2. Découverte de la ressource CSV la plus récente via l'API CKAN de HDX.
3. Repli automatique sur Hugging Face (electricsheepafrica).
4. Repli de secours sur le fichier local existant en cas de coupure réseau.
Sauvegarde dans data/raw/ avec horodatage et maintient une copie 'latest.csv'.
"""

from pathlib import Path
from typing import Optional, Tuple
import datetime
import json
import logging
import shutil
import urllib.request
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/retrain.yaml") -> dict:
    """Charge la configuration YAML du réentraînement."""
    p = Path(config_path)
    if not p.exists():
        alt = Path("benin-food-price-forecast") / config_path
        if alt.exists():
            p = alt
        else:
            p = Path("config/retrain.yaml")
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def download_from_hdx_ckan(package_id: str, ckan_url: str, timeout: int = 25) -> Optional[str]:
    """Interroge l'API CKAN de HDX pour trouver le lien de téléchargement direct du CSV le plus récent."""
    api_endpoint = f"{ckan_url}?id={package_id}"
    req = urllib.request.Request(api_endpoint, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status == 200:
                payload = json.loads(response.read().decode("utf-8"))
                resources = payload.get("result", {}).get("resources", [])
                for r in resources:
                    if r.get("format", "").lower() == "csv" and "download" in r.get("url", "").lower():
                        return r.get("url")
                    if r.get("format", "").lower() == "csv":
                        return r.get("download_url") or r.get("url")
    except Exception as e:
        logger.warning(f"Échec de l'interrogation CKAN HDX ({e})")
    return None


def fetch_latest_food_prices(
    source_type: str = "auto",
    config: Optional[dict] = None,
    output_dir: str = "data/raw",
    force_local: bool = False
) -> Tuple[pd.DataFrame, Path, str]:
    """Télécharge la dernière version des données de prix PAM / HDX.

    Returns
    -------
    df : pd.DataFrame
        Table brute téléchargée.
    saved_path : Path
        Chemin du fichier brut sauvegardé.
    source_used : str
        Source effective ('hdx_direct', 'hdx_ckan', 'huggingface', 'local_fallback').
    """
    if config is None:
        config = load_config()

    src_cfg = config.get("source", {})
    raw_dir = Path(config.get("paths", {}).get("raw_data_dir", output_dir))
    raw_dir.mkdir(parents=True, exist_ok=True)

    today_str = datetime.date.today().strftime("%Y%m%d")
    timestamped_file = raw_dir / f"wfp_food_prices_ben_{today_str}.csv"
    latest_file = raw_dir / "wfp_food_prices_ben_latest.csv"

    if force_local:
        return _fallback_to_local(src_cfg, raw_dir, timestamped_file, latest_file)

    csv_url = src_cfg.get("csv_url")
    ckan_api = src_cfg.get("ckan_api_url")
    pkg_id = src_cfg.get("hdx_package_id", "wfp-food-prices-for-benin")
    timeout = int(src_cfg.get("timeout_seconds", 25))

    # 1. Tentative par URL CSV directe HDX
    if csv_url:
        try:
            logger.info(f"Tentative de téléchargement HDX direct : {csv_url}")
            req = urllib.request.Request(csv_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    content = resp.read()
                    if len(content) > 1000:
                        with open(timestamped_file, "wb") as f:
                            f.write(content)
                        shutil.copyfile(timestamped_file, latest_file)
                        df = pd.read_csv(timestamped_file)
                        logger.info(f"Téléchargement HDX direct réussi ({len(df):,} lignes).")
                        return df, timestamped_file, "hdx_direct"
        except Exception as e:
            logger.warning(f"Échec téléchargement direct HDX : {e}")

    # 2. Tentative par API CKAN HDX
    if ckan_api:
        try:
            logger.info("Recherche de ressource via API CKAN HDX...")
            discovered_url = download_from_hdx_ckan(pkg_id, ckan_api, timeout=timeout)
            if discovered_url and discovered_url != csv_url:
                req = urllib.request.Request(discovered_url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    if resp.status == 200:
                        content = resp.read()
                        if len(content) > 1000:
                            with open(timestamped_file, "wb") as f:
                                f.write(content)
                            shutil.copyfile(timestamped_file, latest_file)
                            df = pd.read_csv(timestamped_file)
                            logger.info(f"Téléchargement via API CKAN réussi ({len(df):,} lignes).")
                            return df, timestamped_file, "hdx_ckan"
        except Exception as e:
            logger.warning(f"Échec via API CKAN : {e}")

    # 3. Tentative de repli Hugging Face
    hf_url = src_cfg.get("huggingface_url")
    if hf_url:
        try:
            logger.info(f"Tentative de téléchargement de secours Hugging Face : {hf_url}")
            req = urllib.request.Request(hf_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    content = resp.read()
                    if len(content) > 1000:
                        with open(timestamped_file, "wb") as f:
                            f.write(content)
                        shutil.copyfile(timestamped_file, latest_file)
                        df = pd.read_csv(timestamped_file)
                        logger.info(f"Téléchargement Hugging Face réussi ({len(df):,} lignes).")
                        return df, timestamped_file, "huggingface"
        except Exception as e:
            logger.warning(f"Échec téléchargement Hugging Face : {e}")

    # 4. Repli de sécurité sur le fichier local
    return _fallback_to_local(src_cfg, raw_dir, timestamped_file, latest_file)


def _fallback_to_local(
    src_cfg: dict,
    raw_dir: Path,
    timestamped_file: Path,
    latest_file: Path
) -> Tuple[pd.DataFrame, Path, str]:
    """Active le repli sur les fichiers locaux en cas d'indisponibilité réseau."""
    logger.warning("Activation du repli sur le stockage local local_fallback...")

    # Vérifier latest.csv
    if latest_file.exists():
        df = pd.read_csv(latest_file)
        logger.info(f"Repli sur {latest_file} réussi ({len(df):,} lignes).")
        return df, latest_file, "local_latest"

    # Vérifier data/train.parquet ou train + test
    candidate_local = src_cfg.get("fallback_local_path", "data/train.parquet")
    p = Path(candidate_local)
    if not p.exists():
        alt = Path("benin-food-price-forecast") / candidate_local
        if alt.exists():
            p = alt

    if p.exists():
        if p.suffix == ".parquet":
            # Recomposer train + test si présent
            p_test = p.parent / "test.parquet"
            if p_test.exists():
                df_tr = pd.read_parquet(p)
                df_te = pd.read_parquet(p_test)
                df = pd.concat([df_tr, df_te], ignore_index=True)
            else:
                df = pd.read_parquet(p)
        else:
            df = pd.read_csv(p)

        df.to_csv(latest_file, index=False)
        df.to_csv(timestamped_file, index=False)
        logger.info(f"Repli local réussi depuis {p} ({len(df):,} lignes sauvegardées dans {latest_file}).")
        return df, latest_file, "local_dataset_fallback"

    raise RuntimeError("Impossible de récupérer les données : échec réseau et aucun fichier local trouvé.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    df, path, src = fetch_latest_food_prices()
    print(f"Téléchargement terminé : {len(df):,} lignes récupérées depuis '{src}' -> {path}")


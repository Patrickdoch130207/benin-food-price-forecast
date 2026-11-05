"""Construction des features pour prévoir le prix d'un produit sur un marché.

Convention : une ligne = une série (produit x marché) et un mois M à prévoir.
La cible `prix` est le prix du mois M. Toutes les features n'utilisent que des
mois STRICTEMENT AVANT M. Une feature à la date M ne dépend donc jamais d'un
prix observé en M ou après.

Usage :
    from src.features import construire_features
    feats = construire_features(pd.read_parquet("data/prix_propres.parquet"))
"""
import numpy as np
import pandas as pd

# Les 4 départements du nord, comme dans 1.4 de 02_eda.ipynb
DEPARTEMENTS_NORD = ["Alibori", "Atakora", "Borgou", "Donga"]

# Nom français des produits, comme dans 02_eda.ipynb
NOMS_PRODUITS = {
    "Cassava meal (gari)": "Gari",
    "Rice (imported)": "Riz importé",
    "Maize (white)": "Maïs blanc",
    "Beans (white)": "Haricot blanc",
    "Tomatoes": "Tomate",
    "Oil (groundnut)": "Huile d'arachide",
    "Onions": "Oignon",
}

# En dessous de ce nombre de séries du produit relevées dans un mois, la médiane
# nationale du mois est trop fragile (mai-juin 2019 : 5 séries en tout, au plus 3
# par produit ; mars 2022 : 0). Ailleurs, un produit a 30 à 45 séries par mois. Voir 1.3.
MIN_SERIES_NATIONAL = 10

# On cherche le dernier prix connu au plus 3 mois en arrière.
MAX_ANCIENNETE = 3

# Il faut au moins 6 relevés passés pour estimer l'écart habituel d'un marché.
MIN_RELEVES_ECART = 6

FEATURES = [
    "prix_m1",
    "prix_m12",
    "prix_dernier_connu",
    "anciennete_dernier_prix",
    "med_nat_m1",
    "var_nat_m1",
    "ecart_marche",
    "mois_annee",
    "latitude",
    "nord",
]


def _long(wide, nom):
    """Passe d'un tableau mois x série à une colonne longue (produit, marché, mois)."""
    return wide.rename_axis(index="mois").unstack().rename(nom)


def construire_features(prix, debut="2019-01", fin=None):
    """Renvoie un tableau (une ligne par série et par mois M avec un prix connu).

    `prix` : le contenu de prix_propres.parquet (colonnes commodity, market,
    date, price, admin1, latitude, longitude).
    """
    prix = prix.copy()
    prix["mois"] = prix["date"].dt.to_period("M")
    fin = fin or prix["mois"].max()
    calendrier = pd.period_range(debut, fin, freq="M")

    # 1. Calendrier mensuel complet : un mois sans relevé devient une case vide
    large = prix.pivot_table(index="mois", columns=["commodity", "market"], values="price")
    large = large.reindex(calendrier)
    large.columns.names = ["commodity", "market"]
    position = np.arange(len(calendrier))

    # 2. Statistiques nationales, produit par produit (médiane des marchés)
    med_nat, var_nat = {}, {}
    for produit in large.columns.get_level_values("commodity").unique():
        w = large[produit]
        med = w.median(axis=1).where(w.notna().sum(axis=1) >= MIN_SERIES_NATIONAL)
        var = w.pct_change(fill_method=None)
        var = var.median(axis=1).where(var.notna().sum(axis=1) >= MIN_SERIES_NATIONAL) * 100
        med_nat[produit], var_nat[produit] = med, var
    med_nat = pd.DataFrame(med_nat, index=calendrier)
    var_nat = pd.DataFrame(var_nat, index=calendrier)

    # 3. Rapport au prix national du mois, puis écart habituel du marché
    #    (moyenne des rapports des mois passés, décalée d'un mois)
    rapport = large.copy()
    for produit in med_nat.columns:
        rapport[produit] = large[produit].div(med_nat[produit], axis=0)
    ecart = rapport.expanding(min_periods=MIN_RELEVES_ECART).mean().shift(1)

    # 4. Dernier prix connu avant M, et son ancienneté en mois
    derniere_pos = pd.DataFrame(
        np.where(large.notna(), position[:, None], np.nan), index=calendrier, columns=large.columns
    ).ffill().shift(1)
    anciennete = pd.DataFrame(position[:, None] - derniere_pos, index=calendrier, columns=large.columns)
    dernier = large.ffill().shift(1).where(anciennete <= MAX_ANCIENNETE)
    anciennete = anciennete.where(anciennete <= MAX_ANCIENNETE)

    # 5. Assemblage en format long
    feats = pd.concat(
        [
            _long(large, "prix"),
            _long(large.shift(1), "prix_m1"),
            _long(large.shift(12), "prix_m12"),
            _long(dernier, "prix_dernier_connu"),
            _long(anciennete, "anciennete_dernier_prix"),
            _long(ecart, "ecart_marche"),
        ],
        axis=1,
    ).reset_index()
    feats = feats.dropna(subset=["prix"])  # on garde les mois où la cible existe

    # Statistiques nationales du mois d'avant, rattachées par (produit, mois M)
    cle = pd.MultiIndex.from_frame(feats[["commodity", "mois"]])
    feats["med_nat_m1"] = med_nat.shift(1).unstack().reindex(cle).to_numpy()
    feats["var_nat_m1"] = var_nat.shift(1).unstack().reindex(cle).to_numpy()

    # 6. Variables qui ne changent pas dans le temps, ou qui ne dépendent que de M
    marches = prix.drop_duplicates("market").set_index("market")[["admin1", "latitude", "longitude"]]
    feats = feats.join(marches, on="market")
    feats["nord"] = feats["admin1"].isin(DEPARTEMENTS_NORD).astype(int)
    feats["mois_annee"] = feats["mois"].dt.month
    feats["annee"] = feats["mois"].dt.year
    feats["produit"] = feats["commodity"].map(NOMS_PRODUITS)
    feats["date"] = feats["mois"].dt.to_timestamp()

    colonnes = ["commodity", "produit", "market", "admin1", "mois", "date", "annee", "prix"] + FEATURES
    return feats[colonnes].sort_values(["commodity", "market", "mois"]).reset_index(drop=True)


def verifier_sans_fuite(prix, feats, mois_test=("2021-06", "2023-03", "2025-01")):
    """Contrôle qu'aucune feature ne voit le futur.

    Pour chaque mois M testé, on multiplie par 7 tous les prix des mois >= M,
    on reconstruit les features, et on compare les lignes du mois M : elles
    doivent être identiques. Renvoie un dict {mois: nombre de colonnes qui diffèrent}.
    """
    resultats = {}
    for m in mois_test:
        m = pd.Period(m, freq="M")
        modif = prix.copy()
        suivants = modif["date"].dt.to_period("M") >= m
        modif.loc[suivants, "price"] = modif.loc[suivants, "price"] * 7
        nouvelles = construire_features(modif)
        a = feats[feats["mois"] == m].set_index(["commodity", "market"])[FEATURES]
        b = nouvelles[nouvelles["mois"] == m].set_index(["commodity", "market"])[FEATURES]
        b = b.reindex(a.index)
        diff = ~((a == b) | (a.isna() & b.isna())).all()
        resultats[str(m)] = int(diff.sum())
    return resultats

## Features (partie 5)

Le fichier `data/features.parquet` est construit par `notebooks/03_features.ipynb`, à partir de `data/prix_propres.parquet`, avec la fonction `construire_features` de `src/features.py`.

**Convention.** Une ligne = une série (produit x marché) et un mois M à prévoir. La cible est `prix` (prix du mois M). Toutes les features n'utilisent que des mois strictement avant M. Une valeur vide veut dire « inconnue » et n'est jamais remplie avec une valeur du futur.

Les features et leur justification sont décrites en 4.3 de `notebooks/02_eda.ipynb` et en 5.4 de `notebooks/03_features.ipynb`.

Ordre d'exécution : `01_cleaning.ipynb` -> `02_eda.ipynb` -> `03_features.ipynb`.

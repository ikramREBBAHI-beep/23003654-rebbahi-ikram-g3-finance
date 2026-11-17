# F8 — Peut-on battre la marche aléatoire ?

Projet d'apprentissage supervisé (classification avec référence, niveau intermédiaire).
Question : des modèles d'apprentissage supervisé prévoient-ils le **sens du rendement du lendemain** de l'indice **MASI** (Bourse de Casablanca, 2018-2025) mieux que la référence « marche aléatoire » (toujours prédire la classe majoritaire) ?

> Groupe : *à compléter* · Encadrant : *à compléter* · Renommer le dossier racine en `CodeApogee-Nom-Groupe`.

## Résultat en une phrase
Non, pas de façon démontrée : sur le test, aucun modèle ne dépasse la référence (50,3 % à 53,4 % contre 58,6 %) ; en validation glissante, le meilleur (forêt aléatoire, 56,1 % contre 53,3 %) n'est pas significatif (McNemar, p = 0,14).

## Arborescence
```
data/raw/        données brutes (MASI, CSV fourni)
data/processed/  table préparée par le notebook (variables + cible)
docs/            fiche du sujet, références
notebooks/       projet_F8.ipynb (final, exécuté, avec sorties)
src/             telecharger_donnees.py (facultatif)
figures/         graphiques exportés par le notebook
rapport/         rapport.pdf (export du notebook) et latex/ (sources)
presentation/    presentation.pdf (6 diapositives) et presentation.pptx
DECLARATION_IA.md
```

## Reproduire
```bash
python -m venv .venv && source .venv/bin/activate      # Windows : .venv\Scripts\activate
pip install -r requirements.txt
cd notebooks && jupyter lab projet_F8.ipynb             # Run All
```
Le notebook lit `../data/raw/Moroccan_All_Shares_Historical_Data__1_.csv`, utilise `GRAINE = 42`, et réécrit `figures/` et `data/processed/`. Versions utilisées : Python 3.12, voir `requirements.txt`.

## Remarques
- Le sujet F8 mentionne « FRED SP500 » ; le jeu fourni est le MASI, utilisé ici. `src/telecharger_donnees.py` permet de récupérer le S&P 500 de FRED (non testé hors ligne) si l'on veut refaire l'étude dessus.
- Format du fichier de type Investing.com : source exacte à confirmer.
- Le rapport PDF est généré depuis le notebook ; les sources LaTeX sont dans `rapport/latex/`.

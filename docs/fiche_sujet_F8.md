# Fiche sujet — F8

| Champ | Contenu |
|---|---|
| Code | F8 |
| Titre | Peut-on battre la marche aléatoire ? |
| Type | Supervisé, classification avec référence |
| Niveau | Intermédiaire |
| Source annoncée | FRED SP500 |
| Source utilisée | Indice MASI, fichier `data/raw/Moroccan_All_Shares_Historical_Data__1_.csv` (1 988 séances, 02/01/2018 → 31/12/2025) |

## Problématique
Si les prix suivent une marche aléatoire, aucun modèle ne peut faire mieux qu'une règle triviale. Des modèles supervisés alimentés par des indicateurs techniques font-ils significativement mieux que cette référence, hors échantillon ?

## Objectif mesurable
Sur le jeu de test chronologique (15 % final), obtenir une précision supérieure à la référence « classe majoritaire du train » avec p < 0,05 (test binomial exact unilatéral). Critères secondaires : précision équilibrée, MCC, AUC, McNemar, validation glissante, rendement après coûts.

## Cible
`cible = 1` si le rendement log de la séance suivante est > 0, sinon 0.

## Modèles et références
Références : classe majoritaire, tirage pondéré, persistance. Modèles : régression logistique (L2, variables standardisées), forêt aléatoire, gradient boosting.

## Protocole
Découpage chronologique 70/15/15 avec embargo d'une ligne ; hyperparamètres choisis sur la validation ; évaluation unique sur le test ; robustesse par walk-forward (2021-2025) ; backtest simplifié avec coût de 0,10 % par changement.

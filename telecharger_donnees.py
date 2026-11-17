"""Téléchargement facultatif de la série FRED SP500 (variante du sujet F8).

Usage : python src/telecharger_donnees.py
Écrit data/raw/SP500_FRED.csv. Nécessite un accès Internet (non testé ici).
Le notebook fourni travaille, lui, sur le MASI déjà présent dans data/raw/.
"""
from pathlib import Path
from urllib.request import urlretrieve

URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500"
DEST = Path(__file__).resolve().parent.parent / "data" / "raw" / "SP500_FRED.csv"


def telecharger(url: str = URL, dest: Path = DEST) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, dest)
    return dest


if __name__ == "__main__":
    print("Fichier écrit :", telecharger())

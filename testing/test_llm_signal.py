import argparse
import json
from pathlib import Path

from signals.llm_signal import analyser_dossier_llm


def charger_dossier(dossier):
    dossier_path = Path(dossier)

    # Si c'est un chemin vers un dossier existant
    if dossier_path.is_dir():
        path = dossier_path / "dossier.json"

    # Si c'est un chemin vers un fichier JSON
    elif dossier_path.is_file():
        path = dossier_path

    # Sinon, on considère que c'est un ID comme D005
    else:
        path = Path("data/json_validated") / dossier / "dossier.json"

    if not path.exists():
        raise FileNotFoundError(
            f"Dossier introuvable : {path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dossier",
        required=True,
        help=(
            "ID du dossier ou chemin vers le dossier de test. "
            "Exemple : D005 ou data/testing/llm_fraud_cases/D9008"
        )
    )

    args = parser.parse_args()

    dossier = args.dossier

    print("=" * 60)
    print("===== TEST SIGNAL LLM =====")
    print("=" * 60)
    print(f"Dossier testé : {dossier}")
    print()

    dossier_json = charger_dossier(dossier)

    resultat = analyser_dossier_llm(dossier_json)

    print(f"Score LLM : {resultat['score']}")
    print(f"Verdict   : {resultat['verdict']}")
    print()

    if resultat["anomalies"]:
        print("Anomalies :")
        for anomalie in resultat["anomalies"]:
            print(f"- {anomalie}")
    else:
        print("Anomalies : aucune")

    print()
    print(f"Explication : {resultat.get('explication', '')}")


if __name__ == "__main__":
    main()
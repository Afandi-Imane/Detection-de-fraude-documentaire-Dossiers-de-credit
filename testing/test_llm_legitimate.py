import json
from pathlib import Path

from signals.llm_signal import analyser_dossier_llm


ROOT = Path("data/json_validated")


def charger_dossier(dossier_id):
    path = ROOT / dossier_id / "dossier.json"

    if not path.exists():
        print(f"[ERREUR] Dossier introuvable : {path}")
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():

    # Dossiers légitimes à tester
    dossiers = [
        "D005",
        "D_TEST",
        "D145_D_TEST",
        "D147_D_TEST",
        "D149_D_TEST",
        "D152_D_TEST",
    ]

    print("=" * 70)
    print("VALIDATION DU SIGNAL LLM SUR DOSSIERS LÉGITIMES")
    print("=" * 70)

    resultats = []

    for dossier_id in dossiers:

        print(f"\n--- {dossier_id} ---")

        dossier = charger_dossier(dossier_id)

        if dossier is None:
            continue

        try:
            resultat = analyser_dossier_llm(dossier)

            score = resultat.get("score", 0)
            verdict = resultat.get("verdict", "")
            anomalies = resultat.get("anomalies", [])

            print(f"Score    : {score}")
            print(f"Verdict  : {verdict}")
            print(f"Anomalies: {len(anomalies)}")

            if anomalies:
                for anomalie in anomalies:
                    print(f"  - {anomalie}")

            resultats.append({
                "dossier_id": dossier_id,
                "score": score,
                "verdict": verdict,
                "anomalies": anomalies
            })

        except Exception as e:
            print(f"[ERREUR LLM] {e}")

    # Résumé
    print("\n" + "=" * 70)
    print("RÉSUMÉ")
    print("=" * 70)

    normaux = sum(
        1 for r in resultats
        if r["verdict"].upper() == "NORMAL"
    )

    suspects = len(resultats) - normaux

    print(f"Nombre de dossiers testés : {len(resultats)}")
    print(f"NORMAL                    : {normaux}")
    print(f"SUSPECT / À SURVEILLER   : {suspects}")

    if resultats:
        taux = normaux / len(resultats) * 100
        print(f"Taux classé NORMAL        : {taux:.1f}%")

    # Sauvegarde
    output = Path("data/testing/llm_legitimate_results.json")

    with open(output, "w", encoding="utf-8") as f:
        json.dump(resultats, f, ensure_ascii=False, indent=2)

    print(f"\nRésultats sauvegardés dans : {output}")


if __name__ == "__main__":
    main()
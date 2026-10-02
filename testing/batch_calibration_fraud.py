from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

FRAUD_ROOT = Path("dossiers_fraude")
PIPELINE_SCRIPT = Path("pipeline_V2.py")

OUTPUT_CSV = Path("data/calibration/fraud_scores_fraud.csv")


# ============================================================
# TROUVER UNIQUEMENT LES DOSSIERS FRAUDULEUX
# ============================================================

def trouver_dossiers_fraude() -> list[Path]:
    """
    Retourne uniquement les dossier.json présents
    dans dossiers_fraude/.
    """

    dossiers = sorted(
        FRAUD_ROOT.glob("*/dossier.json")
    )

    return dossiers


# ============================================================
# LANCER PIPELINE V2
# ============================================================

def lancer_pipeline(dossier_json: Path) -> None:
    """
    Lance pipeline_V2.py sur un dossier frauduleux.
    """

    print(
        f"\n[PIPELINE] {dossier_json.parent.name}"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(PIPELINE_SCRIPT),
            "--input",
            str(dossier_json),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(
            f"[ERREUR] Pipeline échoué pour "
            f"{dossier_json.parent.name}"
        )

        print(result.stderr)

        raise RuntimeError(
            f"Pipeline V2 failed: {dossier_json}"
        )


# ============================================================
# CHERCHER LE SIGNAL LLM
# ============================================================

def extraire_score_llm(dossier: dict) -> float | None:

    signal = dossier.get("signals", {}).get("LLM")

    if isinstance(signal, dict):
        score = signal.get("score")

        if score is not None:
            return float(score)

    return None


# ============================================================
# EXTRAIRE LES SCORES
# ============================================================

def extraire_scores(dossier_json: Path) -> dict:

    with dossier_json.open(
        "r",
        encoding="utf-8",
    ) as f:

        dossier = json.load(f)

    dossier_id = dossier.get(
        "dossier_id",
        dossier_json.parent.name,
    )

    return {
        "dossier_id": dossier_id,
        "label": 1,

        "score_business": dossier.get(
            "score_business"
        ),

        "score_intra": dossier.get(
            "score_intra_dossier"
        ),

        "score_inter": dossier.get(
            "score_inter_dossier"
        ),

        "score_llm": extraire_score_llm(
            dossier
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=========================================="
    )
    print(
        " CALIBRATION DES DOSSIERS FRAUDULEUX"
    )
    print(
        "=========================================="
    )

    # --------------------------------------------------------
    # UNIQUEMENT dossiers_fraude/
    # --------------------------------------------------------

    dossiers = trouver_dossiers_fraude()

    print(
        f"\nDossiers frauduleux trouvés : "
        f"{len(dossiers)}"
    )

    if not dossiers:

        raise RuntimeError(
            f"Aucun dossier trouvé dans "
            f"{FRAUD_ROOT}"
        )

    # --------------------------------------------------------
    # Création du dossier de sortie
    # --------------------------------------------------------

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    # --------------------------------------------------------
    # Traitement
    # --------------------------------------------------------

    for dossier_json in dossiers:

        print(
            f"\n[DOSSIER] "
            f"{dossier_json.parent.name}"
        )

        # 1. Pipeline V2
        lancer_pipeline(
            dossier_json
        )

        # 2. Récupération des scores
        row = extraire_scores(
            dossier_json
        )

        rows.append(row)

    # --------------------------------------------------------
    # Écriture CSV
    # --------------------------------------------------------

    fieldnames = [
        "dossier_id",
        "label",
        "score_business",
        "score_intra",
        "score_inter",
        "score_llm",
    ]

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        "\n=========================================="
    )

    print(
        f"CSV créé : {OUTPUT_CSV}"
    )

    print(
        f"Nombre de fraudes : {len(rows)}"
    )

    print(
        "Les dossiers légitimes n'ont pas été "
        "recalculés."
    )

    print(
        "=========================================="
    )


if __name__ == "__main__":
    main()
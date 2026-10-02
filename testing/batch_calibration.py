from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path
from typing import Any


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

JSON_CORRECTED_DIR = (
    PROJECT_ROOT
    / "data"
    / "json_corrected"
)

SIGNALS_DIR = (
    PROJECT_ROOT
    / "data"
    / "signals"
)

CALIBRATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "calibration"
)

OUTPUT_CSV = (
    CALIBRATION_DIR
    / "legitimate_scores.csv"
)

PIPELINE_V2 = (
    PROJECT_ROOT
    / "pipeline_V2.py"
)


# ============================================================
# UTILITAIRES
# ============================================================

def load_json(path: Path) -> dict[str, Any]:

    import json

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def get_score(
    signal: dict[str, Any] | None,
) -> float | None:

    if not isinstance(signal, dict):
        return None

    score = signal.get("score")

    if score is None:
        return None

    try:
        return float(score)

    except (
        TypeError,
        ValueError,
    ):
        return None


def normalize_score(
    score: float | None,
) -> float | None:

    """
    Sécurise les scores dans [0, 1].

    IMPORTANT :
    cette fonction ne transforme pas un score arbitraire.
    Elle vérifie seulement qu'il est déjà normalisé.
    """

    if score is None:
        return None

    if score < 0 or score > 1:

        print(
            f"[AVERTISSEMENT] "
            f"Score hors [0,1] : {score}"
        )

        return None

    return round(
        score,
        6,
    )


# ============================================================
# LIRE signals_v2.json
# ============================================================

def extract_scores(
    dossier_id: str,
) -> dict[str, Any]:

    signals_json = (
        SIGNALS_DIR
        / dossier_id
        / "signals_v2.json"
    )

    if not signals_json.exists():

        raise FileNotFoundError(
            f"signals_v2.json introuvable : "
            f"{signals_json}"
        )

    data = load_json(
        signals_json
    )

    signals = data.get(
        "signals",
        {},
    )

    business = signals.get(
        "business_rules"
    )

    llm = signals.get(
        "llm"
    )

    intra = signals.get(
        "intra_dossier"
    )

    inter = signals.get(
        "inter_dossiers"
    )

    # --------------------------------------------------------
    # Signal visuel
    #
    # Ton pipeline_V2 actuel ne l'a PAS encore intégré.
    # On laisse donc vide pour le moment.
    # --------------------------------------------------------

    visual = signals.get(
        "visual"
    )

    return {

        "dossier_id": dossier_id,

        "label": 0,

        "fraud_type": "NORMAL",

        "S_business": normalize_score(
            get_score(business)
        ),

        "S_llm": normalize_score(
            get_score(llm)
        ),

        "S_intra": normalize_score(
            get_score(intra)
        ),

        "S_inter": normalize_score(
            get_score(inter)
        ),

        "S_visual": normalize_score(
            get_score(visual)
        ),

        "signals_json": str(
            signals_json.resolve()
        ),
    }


# ============================================================
# LANCER PIPELINE V2
# ============================================================

def run_pipeline_v2(
    dossier_dir: Path,
) -> bool:

    dossier_id = dossier_dir.name

    print()
    print("=" * 70)
    print(
        f" TRAITEMENT {dossier_id}"
    )
    print("=" * 70)

    command = [

        sys.executable,

        str(
            PIPELINE_V2
        ),

        "--input",

        str(
            dossier_dir
        ),

        "--project-root",

        str(
            PROJECT_ROOT
        ),
    ]

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
    )

    if result.returncode != 0:

        print(
            f"[ERREUR] Pipeline V2 échoué "
            f"pour {dossier_id}"
        )

        return False

    print(
        f"[OK] Pipeline V2 terminé "
        f"pour {dossier_id}"
    )

    return True


# ============================================================
# CRÉER CSV
# ============================================================

def save_csv(
    rows: list[dict[str, Any]],
) -> None:

    CALIBRATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [

        "dossier_id",

        "label",

        "fraud_type",

        "S_business",

        "S_llm",

        "S_intra",

        "S_inter",

        "S_visual",

        "signals_json",
    ]

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(rows)

    print()
    print("=" * 70)

    print(
        f"[CSV CRÉÉ] {OUTPUT_CSV}"
    )

    print(
        f"[DOSSIERS] {len(rows)}"
    )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print("=" * 70)
    print(
        " BATCH CALIBRATION - DATASET LÉGITIME"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Vérifications
    # --------------------------------------------------------

    if not JSON_CORRECTED_DIR.exists():

        raise FileNotFoundError(
            "Répertoire json_corrected introuvable : "
            f"{JSON_CORRECTED_DIR}"
        )

    if not PIPELINE_V2.exists():

        raise FileNotFoundError(
            "pipeline_V2.py introuvable : "
            f"{PIPELINE_V2}"
        )

    # --------------------------------------------------------
    # Récupérer tous les dossiers Dxxx
    # --------------------------------------------------------

    dossier_dirs = sorted(

        [
            path
            for path in JSON_CORRECTED_DIR.iterdir()

            if path.is_dir()
            and path.name.startswith("D")
        ],

        key=lambda path: path.name,
    )

    print(
        f"[INFO] "
        f"{len(dossier_dirs)} dossiers trouvés."
    )

    if not dossier_dirs:

        print(
            "[INFO] Aucun dossier à traiter."
        )

        return

    # --------------------------------------------------------
    # Traitement
    # --------------------------------------------------------

    rows: list[
        dict[str, Any]
    ] = []

    failed: list[str] = []

    for index, dossier_dir in enumerate(
        dossier_dirs,
        start=1,
    ):

        dossier_id = dossier_dir.name

        print()
        print(
            f"[{index}/{len(dossier_dirs)}] "
            f"{dossier_id}"
        )

        # ----------------------------------------------------
        # Pipeline V2
        # ----------------------------------------------------

        success = run_pipeline_v2(
            dossier_dir
        )

        if not success:

            failed.append(
                dossier_id
            )

            continue

        # ----------------------------------------------------
        # Récupération signals_v2.json
        # ----------------------------------------------------

        try:

            row = extract_scores(
                dossier_id
            )

            rows.append(
                row
            )

            print(
                "[SCORES] "
                f"business={row['S_business']} | "
                f"llm={row['S_llm']} | "
                f"intra={row['S_intra']} | "
                f"inter={row['S_inter']} | "
                f"visual={row['S_visual']}"
            )

        except Exception as error:

            print(
                f"[ERREUR] "
                f"Impossible de récupérer "
                f"les scores de {dossier_id} : "
                f"{error}"
            )

            failed.append(
                dossier_id
            )

    # --------------------------------------------------------
    # Sauvegarde CSV
    # --------------------------------------------------------

    if rows:

        save_csv(
            rows
        )

    # --------------------------------------------------------
    # Résumé
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        " RÉSUMÉ"
    )
    print("=" * 70)

    print(
        f"Dossiers trouvés : "
        f"{len(dossier_dirs)}"
    )

    print(
        f"Dossiers réussis : "
        f"{len(rows)}"
    )

    print(
        f"Dossiers échoués : "
        f"{len(failed)}"
    )

    if failed:

        print()
        print(
            "Dossiers en erreur :"
        )

        for dossier_id in failed:

            print(
                f"  - {dossier_id}"
            )

    print()
    print(
        "Batch terminé."
    )


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":

    main()
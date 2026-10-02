import random
import shutil
import subprocess
import sys
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_ROOT = Path("data/dossiers_V2")
FRAUD_ROOT = Path("data/dossiers_fraude")

SEED = 42

N_BUSINESS = 25
N_INTRA = 25
N_INTER = 20
N_LLM = 15
N_MIX = 15

GENERATORS = {
    "BUSINESS": "generate_fraud_business.py",
    "INTRA": "generate_fraud_intra.py",
    "INTER": "generate_fraud_inter.py",
    "LLM": "generate_fraud_llm.py",
}

# Types acceptés par les générateurs
BUSINESS_TYPES = [
    "SALAIRE_INVALIDE",
    "ANCIENNETE_INVALIDE",
    "RIB_INVALIDE",
    "IBAN_INVALIDE",
    "DATE_INVALIDE",
]

INTRA_TYPES = [
    "SALAIRE_INCOHERENT",
    "EMPLOYEUR_INCOHERENT",
    "POSTE_INCOHERENT",
    "DATE_INCOHERENTE",
]

INTER_TYPES = [
    "IDENTITE_DIFFERENTE",
    "SALAIRE_DIFFERENT",
    "EMPLOYEUR_DIFFERENT",
    "ADRESSE_DIFFERENTE",
]

LLM_TYPES = [
    "EMPLOYEUR_POSTE",
    "POSTE_SALAIRE",
    "POSTE_ANCIENNETE",
    "PROFIL_PROFESSIONNEL",
]


# ============================================================
# TROUVER LES DOSSIERS LEGITIMES
# ============================================================

def trouver_dossiers():
    return sorted(
        SOURCE_ROOT.glob("*/dossier.json")
    )


# ============================================================
# COPIER LE DOSSIER
# ============================================================

def copier_dossier(source_json, fraud_id):

    destination_dir = FRAUD_ROOT / fraud_id
    destination_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination_json = destination_dir / "dossier.json"

    shutil.copy2(
        source_json,
        destination_json,
    )

    return destination_json


# ============================================================
# APPELER UN GENERATEUR
# ============================================================

def appeler_generateur(
    fraud_type,
    input_json,
    fraud_id,
    specific_type,
):
    """
    Appelle un générateur de fraude.

    Tous les générateurs utilisent :
        --input
        --output-root
        --fraud-type
        --output-id
    """

    generator = GENERATORS[fraud_type]

    script = Path(generator)

    print(
        f"[{fraud_type}] "
        f"{input_json.parent.name} -> {fraud_id} "
        f"({specific_type})"
    )

    command = [
        sys.executable,
        str(script),

        "--input",
        str(input_json),

        "--output-root",
        str(FRAUD_ROOT),

        "--fraud-type",
        specific_type,

        "--output-id",
        fraud_id,
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:

        print(
            f"[ERREUR] {fraud_type} "
            f"{fraud_id}"
        )

        if result.stderr:
            print(result.stderr)

        return False

    if result.stdout:
        print(result.stdout)

    return True


# ============================================================
# CREER UNE FRAUDE SIMPLE
# ============================================================

def creer_fraude_simple(
    source_json,
    fraud_id,
    fraud_type,
    specific_type,
):

    # Le générateur prend directement
    # le dossier légitime comme input.
    return appeler_generateur(
        fraud_type,
        source_json,
        fraud_id,
        specific_type,
    )


# ============================================================
# CREER UNE FRAUDE MIXTE
# ============================================================

def creer_fraude_mixte(
    source_json,
    fraud_id,
    types,
    mix_index,
):

    print(
        f"\n[MIX] {fraud_id} -> "
        f"{' + '.join(types)}"
    )

    # --------------------------------------------------------
    # 1. Première fraude
    # --------------------------------------------------------

    first_type = types[0]

    if first_type == "BUSINESS":
        specific = BUSINESS_TYPES[
            mix_index % len(BUSINESS_TYPES)
        ]

    elif first_type == "INTRA":
        specific = INTRA_TYPES[
            mix_index % len(INTRA_TYPES)
        ]

    elif first_type == "INTER":
        specific = INTER_TYPES[
            mix_index % len(INTER_TYPES)
        ]

    else:
        specific = LLM_TYPES[
            mix_index % len(LLM_TYPES)
        ]

    ok = creer_fraude_simple(
        source_json,
        fraud_id,
        first_type,
        specific,
    )

    if not ok:
        return False

    # --------------------------------------------------------
    # 2. Fraudes suivantes
    # --------------------------------------------------------

    current_json = (
        FRAUD_ROOT
        / fraud_id
        / "dossier.json"
    )

    for j, fraud_type in enumerate(types[1:], start=1):

        if fraud_type == "BUSINESS":
            specific = BUSINESS_TYPES[
                (mix_index + j)
                % len(BUSINESS_TYPES)
            ]

        elif fraud_type == "INTRA":
            specific = INTRA_TYPES[
                (mix_index + j)
                % len(INTRA_TYPES)
            ]

        elif fraud_type == "INTER":
            specific = INTER_TYPES[
                (mix_index + j)
                % len(INTER_TYPES)
            ]

        else:
            specific = LLM_TYPES[
                (mix_index + j)
                % len(LLM_TYPES)
            ]

        ok = appeler_generateur(
            fraud_type,
            current_json,
            fraud_id,
            specific,
        )

        if not ok:
            return False

        current_json = (
            FRAUD_ROOT
            / fraud_id
            / "dossier.json"
        )

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    random.seed(SEED)

    FRAUD_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    dossiers = trouver_dossiers()

    print(
        f"Dossiers légitimes trouvés : "
        f"{len(dossiers)}"
    )

    if len(dossiers) < 230:
        raise RuntimeError(
            f"Il faut au moins 230 dossiers. "
            f"Trouvés : {len(dossiers)}"
        )

    # ========================================================
    # SELECTION DE 100 DOSSIERS DIFFERENTS
    # ========================================================

    selected = random.sample(
        dossiers,
        100,
    )

    random.shuffle(selected)

    index = 0

    # ========================================================
    # 15 LLM
    # ========================================================

    print("\n========== LLM ==========")

    for i in range(N_LLM):

        source = selected[index]

        fraud_id = f"F{index + 1:03d}"

        specific_type = LLM_TYPES[
            i % len(LLM_TYPES)
        ]

        creer_fraude_simple(
            source,
            fraud_id,
            "LLM",
            specific_type,
        )

        index += 1

    # ========================================================
    # 25 BUSINESS
    # ========================================================

    print("\n========== BUSINESS ==========")

    for i in range(N_BUSINESS):

        source = selected[index]

        fraud_id = f"F{index + 1:03d}"

        specific_type = BUSINESS_TYPES[
            i % len(BUSINESS_TYPES)
        ]

        creer_fraude_simple(
            source,
            fraud_id,
            "BUSINESS",
            specific_type,
        )

        index += 1

    # ========================================================
    # 25 INTRA
    # ========================================================

    print("\n========== INTRA ==========")

    for i in range(N_INTRA):

        source = selected[index]

        fraud_id = f"F{index + 1:03d}"

        specific_type = INTRA_TYPES[
            i % len(INTRA_TYPES)
        ]

        creer_fraude_simple(
            source,
            fraud_id,
            "INTRA",
            specific_type,
        )

        index += 1

    # ========================================================
    # 20 INTER
    # ========================================================

    print("\n========== INTER ==========")

    for i in range(N_INTER):

        source = selected[index]

        fraud_id = f"F{index + 1:03d}"

        specific_type = INTER_TYPES[
            i % len(INTER_TYPES)
        ]

        creer_fraude_simple(
            source,
            fraud_id,
            "INTER",
            specific_type,
        )

        index += 1

    # ========================================================
    # 15 MIXTES
    # ========================================================

    print("\n========== MIX ==========")

    mix_types = [
        ["BUSINESS", "INTRA"],
        ["BUSINESS", "INTER"],
        ["INTRA", "INTER"],
        ["BUSINESS", "LLM"],
        ["INTRA", "LLM"],
        ["BUSINESS", "INTRA", "INTER"],
    ]

    for i in range(N_MIX):

        source = selected[index]

        fraud_id = f"F{index + 1:03d}"

        types = mix_types[
            i % len(mix_types)
        ]

        creer_fraude_mixte(
            source,
            fraud_id,
            types,
            i,
        )

        index += 1

    # ========================================================
    # FIN
    # ========================================================

    print()
    print("==============================")
    print("GENERATION TERMINEE")
    print("==============================")

    print(
        f"Fraudes demandées : {index}"
    )

    print(
        f"Sortie : {FRAUD_ROOT}"
    )


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":
    main()
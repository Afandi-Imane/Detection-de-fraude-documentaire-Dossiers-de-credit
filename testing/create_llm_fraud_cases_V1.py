import json
import copy
from pathlib import Path

SOURCE = Path("data/json_validated/D005/dossier.json")
OUTPUT_DIR = Path("data/testing/llm_fraud_cases")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def set_field(dossier, doc_id, field_name, value):
    """Modifie un champ dans document_fields."""
    field = dossier["document_fields"][doc_id][field_name]

    field["raw_value"] = value
    field["normalized_value"] = value


def create_case(case_id, description, modifier):
    with open(SOURCE, "r", encoding="utf-8") as f:
        original = json.load(f)

    dossier = copy.deepcopy(original)
    dossier["dossier_id"] = case_id

    modifier(dossier)

    output = OUTPUT_DIR / f"{case_id}.json"

    with open(output, "w", encoding="utf-8") as f:
        json.dump(dossier, f, ensure_ascii=False, indent=2)

    print(f"[CRÉÉ] {case_id} -> {description}")


# ============================================================
# DOC IDS
# ============================================================

BULLETIN_M1 = "D005_D005_DOC0047_bulletin_M1"
ATTESTATION = "D005_D005_DOC0050_attestation"


# ============================================================
# CAS 9008
# Employeur / poste incohérent
# ============================================================

def case_9008(d):
    # ORANGE MAROC + PHARMACIEN
    set_field(
        d,
        BULLETIN_M1,
        "poste",
        "PHARMACIEN"
    )

    set_field(
        d,
        ATTESTATION,
        "poste",
        "PHARMACIEN"
    )


# ============================================================
# CAS 9009
# Salaire incohérent avec le poste
# ============================================================

def case_9009(d):
    # Ingénieur Télécom avec salaire extrêmement faible
    set_field(
        d,
        BULLETIN_M1,
        "net_a_payer",
        1000.0
    )


# ============================================================
# CAS 9010
# Poste incohérent avec l'employeur
# ============================================================

def case_9010(d):
    # Banque + poste pharmacien
    set_field(
        d,
        BULLETIN_M1,
        "employeur",
        "ATTIJARIWAFA BANK"
    )

    set_field(
        d,
        BULLETIN_M1,
        "poste",
        "PHARMACIEN"
    )


# ============================================================
# CAS 9011
# Profil professionnel incohérent
# ============================================================

def case_9011(d):
    # Employeur télécom + poste totalement différent
    set_field(
        d,
        BULLETIN_M1,
        "poste",
        "MEDECIN CHIRURGIEN"
    )


# ============================================================
# CAS 9012
# Salaire très élevé pour le poste déclaré
# ============================================================

def case_9012(d):
    set_field(
        d,
        BULLETIN_M1,
        "net_a_payer",
        100000.0
    )


# ============================================================
# CRÉATION
# ============================================================

create_case(
    "D9008",
    "Employeur ORANGE MAROC + poste PHARMACIEN",
    case_9008
)

create_case(
    "D9009",
    "INGENIEUR TELECOM + salaire 1000",
    case_9009
)

create_case(
    "D9010",
    "ATTIJARIWAFA BANK + PHARMACIEN",
    case_9010
)

create_case(
    "D9011",
    "ORANGE MAROC + MEDECIN CHIRURGIEN",
    case_9011
)

create_case(
    "D9012",
    "INGENIEUR TELECOM + salaire 100000",
    case_9012
)


print()
print(f"Cas créés dans : {OUTPUT_DIR}")
print("Le dossier D005 original n'a pas été modifié.")
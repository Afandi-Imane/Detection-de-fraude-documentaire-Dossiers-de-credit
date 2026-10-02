import json
from pathlib import Path
from copy import deepcopy


# ============================================================
# CHEMINS
# ============================================================

SOURCE = Path("data/json_validated/D005/dossier.json")
OUTPUT_DIR = Path("data/testing/llm_fraud_cases")


# ============================================================
# CHARGEMENT
# ============================================================

def charger_dossier_source():
    if not SOURCE.exists():
        raise FileNotFoundError(
            f"Dossier source introuvable : {SOURCE}"
        )

    with open(SOURCE, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# ACCÈS AUX DOCUMENTS
# ============================================================

def get_document(dossier, doc_type, index=0):
    """
    Récupère les champs d'un document à partir de document_fields.
    """
    doc_ids = dossier["documents_by_type"][doc_type]

    if index >= len(doc_ids):
        raise IndexError(
            f"Document {doc_type}[{index}] introuvable."
        )

    doc_id = doc_ids[index]

    return dossier["document_fields"][doc_id]


# ============================================================
# MODIFICATION D'UN CHAMP
# ============================================================

def set_field(dossier, doc_type, field_name, value, index=0):
    document = get_document(dossier, doc_type, index)

    if field_name not in document:
        raise KeyError(
            f"Champ '{field_name}' absent de {doc_type}[{index}]"
        )

    document[field_name]["raw_value"] = value
    document[field_name]["normalized_value"] = value

    # On garde une confiance élevée car la fraude est volontairement
    # injectée dans une donnée déjà extraite.
    document[field_name]["confidence"] = 1.0
    document[field_name]["validation_status"] = "VALID"


# ============================================================
# CRÉATION D'UN CAS
# ============================================================

def create_case(dossier_id, description, modifier):
    dossier_source = charger_dossier_source()

    dossier = deepcopy(dossier_source)

    dossier["dossier_id"] = dossier_id

    modifier(dossier)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / f"{dossier_id}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            dossier,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"[CRÉÉ] {dossier_id} -> {description}")


# ============================================================
# D9008
# EMPLOYEUR ↔ POSTE
# ============================================================

def case_9008(d):
    """
    ORANGE MAROC + PHARMACIEN

    Objectif :
    détecter une incohérence sémantique entre
    l'activité de l'employeur et le métier déclaré.
    """

    # Les 3 bulletins sont modifiés pour éviter une simple
    # incohérence temporelle M1 -> M2 -> M3.
    for i in range(3):
        set_field(
            d,
            "BULLETIN_SALAIRE",
            "poste",
            "PHARMACIEN",
            index=i
        )

    # L'attestation doit également correspondre.
    set_field(
        d,
        "ATTESTATION_TRAVAIL",
        "poste",
        "PHARMACIEN"
    )


# ============================================================
# D9009
# POSTE ↔ SALAIRE TRÈS FAIBLE
# ============================================================

def case_9009(d):
    """
    INGENIEUR TELECOM + salaire de 1000 DH

    Objectif :
    détecter une combinaison poste/salaire très peu plausible.
    """

    for i in range(3):
        set_field(
            d,
            "BULLETIN_SALAIRE",
            "net_a_payer",
            1000.0,
            index=i
        )


# ============================================================
# D9010
# POSTE ↔ SALAIRE TRÈS ÉLEVÉ
# ============================================================

def case_9010(d):
    """
    INGENIEUR TELECOM + salaire de 100000 DH

    Objectif :
    détecter un salaire extrêmement élevé par rapport
    au profil professionnel présenté.

    Les 3 bulletins ont le même salaire afin de ne pas créer
    artificiellement une incohérence temporelle.
    """

    for i in range(3):
        set_field(
            d,
            "BULLETIN_SALAIRE",
            "salaire_base_mensuel",
            100000.0,
            index=i
        )

        set_field(
            d,
            "BULLETIN_SALAIRE",
            "salaire_brut",
            100000.0,
            index=i
        )

        set_field(
            d,
            "BULLETIN_SALAIRE",
            "net_a_payer",
            100000.0,
            index=i
        )


# ============================================================
# D9011
# EMPLOYEUR ↔ POSTE ↔ ANCIENNETÉ
# ============================================================

def case_9011(d):
    """
    ORANGE MAROC + PHARMACIEN + moins de 2 ans

    Objectif :
    tester la cohérence globale du profil professionnel.

    Ici on ne touche PAS au salaire.
    """

    for i in range(3):
        set_field(
            d,
            "BULLETIN_SALAIRE",
            "poste",
            "PHARMACIEN",
            index=i
        )

        set_field(
            d,
            "BULLETIN_SALAIRE",
            "anciennete",
            "MOINS DE 2 ANS",
            index=i
        )

    set_field(
        d,
        "ATTESTATION_TRAVAIL",
        "poste",
        "PHARMACIEN"
    )


# ============================================================
# D9012
# INCOHÉRENCE GLOBALE
# ============================================================

def case_9012(d):
    """
    ORANGE MAROC
    + PHARMACIEN
    + salaire élevé
    + faible ancienneté

    Objectif :
    tester une combinaison de plusieurs éléments
    qui peuvent sembler individuellement possibles,
    mais deviennent peu cohérents ensemble.
    """

    for i in range(3):
        set_field(
            d,
            "BULLETIN_SALAIRE",
            "poste",
            "PHARMACIEN",
            index=i
        )

        set_field(
            d,
            "BULLETIN_SALAIRE",
            "anciennete",
            "MOINS DE 2 ANS",
            index=i
        )

        set_field(
            d,
            "BULLETIN_SALAIRE",
            "salaire_base_mensuel",
            50000.0,
            index=i
        )

        set_field(
            d,
            "BULLETIN_SALAIRE",
            "salaire_brut",
            50000.0,
            index=i
        )

        set_field(
            d,
            "BULLETIN_SALAIRE",
            "net_a_payer",
            50000.0,
            index=i
        )

    set_field(
        d,
        "ATTESTATION_TRAVAIL",
        "poste",
        "PHARMACIEN"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # Créer le dossier de sortie
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Supprimer uniquement les anciens cas de test LLM
    for old_file in OUTPUT_DIR.glob("*.json"):
        old_file.unlink()

    # Créer les nouveaux cas
    create_case(
        "D9008",
        "ORANGE MAROC + poste PHARMACIEN",
        case_9008
    )

    create_case(
        "D9009",
        "INGENIEUR TELECOM + salaire très faible",
        case_9009
    )

    create_case(
        "D9010",
        "INGENIEUR TELECOM + salaire très élevé",
        case_9010
    )

    create_case(
        "D9011",
        "ORANGE MAROC + PHARMACIEN + faible ancienneté",
        case_9011
    )

    create_case(
        "D9012",
        "ORANGE MAROC + PHARMACIEN + 50000 DH + faible ancienneté",
        case_9012
    )

    print()
    print(f"Cas créés dans : {OUTPUT_DIR}")
    print("Le dossier D005 original n'a pas été modifié.")
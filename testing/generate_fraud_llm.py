import copy
import json
from pathlib import Path
from typing import Any


# ============================================================
# FRAUDES LLM POSSIBLES
# ============================================================

FRAUD_TYPES = {
    "EMPLOYEUR_POSTE": {
        "description": "Créer une incohérence sémantique entre l'employeur et le poste."
    },
    "POSTE_SALAIRE": {
        "description": "Créer une combinaison poste + salaire sémantiquement atypique."
    },
    "POSTE_ANCIENNETE": {
        "description": "Créer une combinaison poste + ancienneté atypique."
    },
    "PROFIL_PROFESSIONNEL": {
        "description": "Créer une incohérence dans le profil professionnel."
    },
}


# ============================================================
# OUTILS
# ============================================================

def get_value(field: Any):
    """
    Récupère la valeur d'un champ.
    Compatible avec :
        "champ": "valeur"
    ou :
        "champ": {"normalized_value": "valeur"}
    """
    if isinstance(field, dict):
        return field.get("normalized_value", field.get("value"))

    return field


def set_value(fields: dict, field_name: str, value: Any):
    """
    Modifie uniquement la copie du dossier.
    """

    if field_name not in fields:
        return False

    original = fields[field_name]

    if isinstance(original, dict):
        if "normalized_value" in original:
            original["normalized_value"] = value

        if "value" in original:
            original["value"] = value
    else:
        fields[field_name] = value

    return True


def find_documents(dossier: dict):
    """
    Retourne les documents du dossier.
    """

    return dossier.get("document_fields", {})


# ============================================================
# FRAUDE 1 : EMPLOYEUR ↔ POSTE
# ============================================================

def fraud_employeur_poste(dossier: dict) -> bool:

    documents = find_documents(dossier)

    for doc_id, fields in documents.items():

        if not isinstance(fields, dict):
            continue

        if "poste" not in fields:
            continue

        # On ne modifie pas l'employeur.
        # On modifie seulement le poste.
        set_value(
            fields,
            "poste",
            "PHARMACIEN"
        )

        return True

    return False


# ============================================================
# FRAUDE 2 : POSTE ↔ SALAIRE
# ============================================================

def fraud_poste_salaire(dossier: dict) -> bool:

    documents = find_documents(dossier)

    for doc_id, fields in documents.items():

        if not isinstance(fields, dict):
            continue

        if "poste" not in fields:
            continue

        # Exemple volontairement atypique.
        set_value(
            fields,
            "poste",
            "DIRECTEUR GENERAL"
        )

        if "salaire_base_mensuel" in fields:
            set_value(
                fields,
                "salaire_base_mensuel",
                2500
            )

        elif "net_a_payer" in fields:
            set_value(
                fields,
                "net_a_payer",
                2500
            )

        else:
            continue

        return True

    return False


# ============================================================
# FRAUDE 3 : POSTE ↔ ANCIENNETÉ
# ============================================================

def fraud_poste_anciennete(dossier: dict) -> bool:

    documents = find_documents(dossier)

    for doc_id, fields in documents.items():

        if not isinstance(fields, dict):
            continue

        if "poste" not in fields:
            continue

        if "anciennete" not in fields:
            continue

        set_value(
            fields,
            "poste",
            "DIRECTEUR GENERAL"
        )

        set_value(
            fields,
            "anciennete",
            "1 mois"
        )

        return True

    return False


# ============================================================
# FRAUDE 4 : PROFIL PROFESSIONNEL
# ============================================================

def fraud_profil_professionnel(dossier: dict) -> bool:

    documents = find_documents(dossier)

    modified = False

    for doc_id, fields in documents.items():

        if not isinstance(fields, dict):
            continue

        if "poste" in fields:

            # On modifie seulement un document.
            # Les autres documents gardent le poste original.
            set_value(
                fields,
                "poste",
                "PHARMACIEN"
            )

            modified = True
            break

    return modified


# ============================================================
# APPLICATION DE LA FRAUDE
# ============================================================

def apply_llm_fraud(
    dossier: dict,
    fraud_type: str
) -> dict:

    # IMPORTANT :
    # copie profonde => le JSON original reste intact.
    fraud_dossier = copy.deepcopy(dossier)

    fraud_dossier["fraud"] = {
        "is_fraud": True,
        "fraud_family": "LLM",
        "fraud_type": fraud_type,
    }

    if fraud_type == "EMPLOYEUR_POSTE":
        success = fraud_employeur_poste(fraud_dossier)

    elif fraud_type == "POSTE_SALAIRE":
        success = fraud_poste_salaire(fraud_dossier)

    elif fraud_type == "POSTE_ANCIENNETE":
        success = fraud_poste_anciennete(fraud_dossier)

    elif fraud_type == "PROFIL_PROFESSIONNEL":
        success = fraud_profil_professionnel(fraud_dossier)

    else:
        raise ValueError(
            f"Type de fraude LLM inconnu : {fraud_type}"
        )

    if not success:
        raise RuntimeError(
            f"Impossible d'injecter la fraude {fraud_type} "
            f"dans le dossier {dossier.get('dossier_id')}"
        )

    return fraud_dossier


# ============================================================
# GENERATION D'UN DOSSIER FRAUDULEUX
# ============================================================

def generate_fraud_llm(
    input_path: str | Path,
    output_root: str | Path,
    fraud_type: str,
    output_id: str | None = None,
):

    input_path = Path(input_path)
    output_root = Path(output_root)

    # ----------------------------
    # Lire le dossier légitime
    # ----------------------------

    with input_path.open(
        "r",
        encoding="utf-8"
    ) as file:

        dossier = json.load(file)

    # ----------------------------
    # Créer la copie frauduleuse
    # ----------------------------

    fraud_dossier = apply_llm_fraud(
        dossier,
        fraud_type
    )

    # ----------------------------
    # Nouvel ID
    # ----------------------------

    original_id = dossier.get(
        "dossier_id",
        input_path.parent.name
    )

    fraud_id = output_id or f"{original_id}_LLM"

    fraud_dossier["dossier_id"] = fraud_id

    # ----------------------------
    # Créer :
    #
    # dossiers_fraude/
    #    D231/
    #       dossier.json
    # ----------------------------

    output_dir = output_root / fraud_id

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = output_dir / "dossier.json"

    # ----------------------------
    # Sauvegarde
    # ----------------------------

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            fraud_dossier,
            file,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"[OK] Fraude LLM créée : {output_path}"
    )

    return output_path


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Génère une copie frauduleuse LLM d'un dossier légitime."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Chemin vers dossier.json légitime"
    )

    parser.add_argument(
        "--output-root",
        default="dossiers_fraude",
        help="Dossier de sortie"
    )

    parser.add_argument(
        "--fraud-type",
        required=True,
        choices=list(FRAUD_TYPES.keys()),
        help="Type de fraude LLM"
    )

    parser.add_argument(
        "--output-id",
        required=True,
        help="ID du nouveau dossier frauduleux"
    )

    args = parser.parse_args()

    generate_fraud_llm(
        input_path=args.input,
        output_root=args.output_root,
        fraud_type=args.fraud_type,
        output_id=args.output_id,
    )
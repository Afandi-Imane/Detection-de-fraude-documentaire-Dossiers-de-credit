import copy
import json
from pathlib import Path
from typing import Any


# ============================================================
# OUTILS
# ============================================================

def get_value(field: Any):
    if isinstance(field, dict):
        return field.get("normalized_value", field.get("value"))
    return field


def set_value(field: Any, value: Any):
    if isinstance(field, dict):
        if "normalized_value" in field:
            field["normalized_value"] = value

        if "value" in field:
            field["value"] = value

    return value


def find_field(
    document_fields: dict,
    field_names: list[str],
):
    """
    Cherche un champ dans les documents du dossier.
    Retourne (document_id, field_name, field).
    """

    for doc_id, fields in document_fields.items():

        if not isinstance(fields, dict):
            continue

        for field_name in field_names:

            if field_name in fields:
                return doc_id, field_name, fields[field_name]

    return None, None, None


# ============================================================
# FRAUDES INTRA-DOSSIER
# ============================================================

def fraud_modifier_cin(dossier: dict) -> bool:
    """
    Modifie le CIN dans un seul document.
    """

    document_fields = dossier.get("document_fields", {})

    doc_id, field_name, field = find_field(
        document_fields,
        ["cin", "cin_numero"],
    )

    if field is None:
        return False

    old_value = get_value(field)

    if not old_value:
        return False

    # Exemple de modification déterministe
    new_value = str(old_value) + "X"

    set_value(field, new_value)

    return True


def fraud_modifier_nom(dossier: dict) -> bool:
    """
    Modifie le nom dans un seul document.
    """

    document_fields = dossier.get("document_fields", {})

    doc_id, field_name, field = find_field(
        document_fields,
        ["nom", "nom_employe"],
    )

    if field is None:
        return False

    old_value = get_value(field)

    if not old_value:
        return False

    new_value = f"{old_value} X"

    set_value(field, new_value)

    return True


def fraud_modifier_adresse(dossier: dict) -> bool:
    """
    Modifie l'adresse dans un seul document.
    """

    document_fields = dossier.get("document_fields", {})

    doc_id, field_name, field = find_field(
        document_fields,
        ["adresse"],
    )

    if field is None:
        return False

    old_value = get_value(field)

    if not old_value:
        return False

    new_value = f"{old_value} LOT 99"

    set_value(field, new_value)

    return True


def fraud_modifier_employeur(dossier: dict) -> bool:
    """
    Modifie l'employeur dans un seul document.
    """

    document_fields = dossier.get("document_fields", {})

    doc_id, field_name, field = find_field(
        document_fields,
        ["employeur"],
    )

    if field is None:
        return False

    old_value = get_value(field)

    if not old_value:
        return False

    new_value = f"{old_value} SARL"

    set_value(field, new_value)

    return True


def fraud_modifier_poste(dossier: dict) -> bool:
    """
    Modifie le poste dans un seul document.
    """

    document_fields = dossier.get("document_fields", {})

    doc_id, field_name, field = find_field(
        document_fields,
        ["poste"],
    )

    if field is None:
        return False

    old_value = get_value(field)

    if not old_value:
        return False

    # Exemple : changement de poste
    new_value = "DIRECTEUR"

    set_value(field, new_value)

    return True


def fraud_modifier_date_embauche(dossier: dict) -> bool:
    """
    Modifie la date d'embauche dans un seul document.
    """

    document_fields = dossier.get("document_fields", {})

    doc_id, field_name, field = find_field(
        document_fields,
        ["date_embauche"],
    )

    if field is None:
        return False

    old_value = get_value(field)

    if not old_value:
        return False

    # Exemple de modification
    new_value = "2020-01-01"

    set_value(field, new_value)

    return True


# ============================================================
# APPLICATION
# ============================================================

def apply_intra_fraud(
    dossier: dict,
    fraud_type: str,
) -> dict:

    # IMPORTANT :
    # On ne touche JAMAIS au dossier original.
    fraud_dossier = copy.deepcopy(dossier)

    success = False

    if fraud_type == "MODIFICATION_CIN":
        success = fraud_modifier_cin(fraud_dossier)

    elif fraud_type == "MODIFICATION_NOM":
        success = fraud_modifier_nom(fraud_dossier)

    elif fraud_type == "MODIFICATION_ADRESSE":
        success = fraud_modifier_adresse(fraud_dossier)

    elif fraud_type == "MODIFICATION_EMPLOYEUR":
        success = fraud_modifier_employeur(fraud_dossier)

    elif fraud_type == "MODIFICATION_POSTE":
        success = fraud_modifier_poste(fraud_dossier)

    elif fraud_type == "MODIFICATION_DATE_EMBAUCHE":
        success = fraud_modifier_date_embauche(fraud_dossier)

    else:
        raise ValueError(
            f"Type de fraude intra inconnu : {fraud_type}"
        )

    if not success:
        raise RuntimeError(
            f"Impossible d'injecter la fraude {fraud_type}"
        )

    # Traçabilité
    fraud_dossier["fraud"] = {
        "is_fraud": True,
        "fraud_family": "INTRA",
        "fraud_type": fraud_type,
    }

    return fraud_dossier


# ============================================================
# GENERATION
# ============================================================

def generate_fraud_intra(
    input_path: str | Path,
    output_root: str | Path,
    fraud_type: str,
    output_id: str,
):

    input_path = Path(input_path)
    output_root = Path(output_root)

    # --------------------------------------------------------
    # Lire le dossier légitime
    # --------------------------------------------------------

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        dossier = json.load(file)

    # --------------------------------------------------------
    # Créer la copie frauduleuse
    # --------------------------------------------------------

    fraud_dossier = apply_intra_fraud(
        dossier=dossier,
        fraud_type=fraud_type,
    )

    # Nouveau ID
    fraud_dossier["dossier_id"] = output_id

    # --------------------------------------------------------
    # Sauvegarder
    # --------------------------------------------------------

    output_dir = output_root / output_id

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_dir / "dossier.json"

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            fraud_dossier,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"[OK] Fraude INTRA créée : {output_path}"
    )

    return output_path


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Génère une fraude intra-dossier."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Dossier légitime source",
    )

    parser.add_argument(
        "--output-root",
        default="dossiers_fraude",
        help="Dossier de sortie",
    )

    parser.add_argument(
        "--fraud-type",
        required=True,
        choices=[
            "MODIFICATION_CIN",
            "MODIFICATION_NOM",
            "MODIFICATION_ADRESSE",
            "MODIFICATION_EMPLOYEUR",
            "MODIFICATION_POSTE",
            "MODIFICATION_DATE_EMBAUCHE",
        ],
    )

    parser.add_argument(
        "--output-id",
        required=True,
    )

    args = parser.parse_args()

    generate_fraud_intra(
        input_path=args.input,
        output_root=args.output_root,
        fraud_type=args.fraud_type,
        output_id=args.output_id,
    )
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


def find_field(document_fields: dict, field_names: list[str]):
    for doc_id, fields in document_fields.items():

        if not isinstance(fields, dict):
            continue

        for field_name in field_names:
            if field_name in fields:
                return doc_id, field_name, fields[field_name]

    return None, None, None


# ============================================================
# BUSINESS RULES
# ============================================================

def fraud_salaire_invalide(dossier: dict) -> bool:
    """
    Met un salaire anormalement bas/élevé
    par rapport à la règle métier définie.
    """

    document_fields = dossier.get("document_fields", {})

    _, _, field = find_field(
        document_fields,
        [
            "salaire_base_mensuel",
            "salaire_brut",
            "salaire_net_imposable",
            "net_a_payer",
        ],
    )

    if field is None:
        return False

    # Valeur volontairement frauduleuse
    new_salary = 500

    set_value(field, new_salary)

    return True


def fraud_anciennete_invalide(dossier: dict) -> bool:
    """
    Crée une ancienneté incohérente avec la date d'embauche.
    """

    document_fields = dossier.get("document_fields", {})

    _, _, field = find_field(
        document_fields,
        ["anciennete"],
    )

    if field is None:
        return False

    set_value(field, "25 ans")

    return True


def fraud_rib_invalide(dossier: dict) -> bool:
    """
    Remplace le RIB par une valeur ne respectant pas
    la structure attendue.
    """

    document_fields = dossier.get("document_fields", {})

    _, _, field = find_field(
        document_fields,
        ["rib"],
    )

    if field is None:
        return False

    set_value(
        field,
        "000000000000000000000000"
    )

    return True


def fraud_iban_invalide(dossier: dict) -> bool:
    """
    Crée un IBAN invalide.
    """

    document_fields = dossier.get("document_fields", {})

    _, _, field = find_field(
        document_fields,
        ["iban"],
    )

    if field is None:
        return False

    set_value(
        field,
        "MA00INVALIDIBAN000000000000"
    )

    return True


def fraud_date_invalide(dossier: dict) -> bool:
    """
    Crée une date métier impossible / incohérente.
    """

    document_fields = dossier.get("document_fields", {})

    _, _, field = find_field(
        document_fields,
        ["date_embauche"],
    )

    if field is None:
        return False

    set_value(
        field,
        "1900-01-01"
    )

    return True


# ============================================================
# APPLICATION
# ============================================================

def apply_business_fraud(
    dossier: dict,
    fraud_type: str,
) -> dict:

    # Toujours travailler sur une COPIE
    fraud_dossier = copy.deepcopy(dossier)

    success = False

    if fraud_type == "SALAIRE_INVALIDE":
        success = fraud_salaire_invalide(fraud_dossier)

    elif fraud_type == "ANCIENNETE_INVALIDE":
        success = fraud_anciennete_invalide(fraud_dossier)

    elif fraud_type == "RIB_INVALIDE":
        success = fraud_rib_invalide(fraud_dossier)

    elif fraud_type == "IBAN_INVALIDE":
        success = fraud_iban_invalide(fraud_dossier)

    elif fraud_type == "DATE_INVALIDE":
        success = fraud_date_invalide(fraud_dossier)

    else:
        raise ValueError(
            f"Type de fraude BUSINESS inconnu : {fraud_type}"
        )

    if not success:
        raise RuntimeError(
            f"Impossible d'injecter la fraude {fraud_type}"
        )

    fraud_dossier["fraud"] = {
        "is_fraud": True,
        "fraud_family": "BUSINESS_RULE",
        "fraud_type": fraud_type,
    }

    return fraud_dossier


# ============================================================
# GENERATION
# ============================================================

def generate_fraud_business(
    input_path: str | Path,
    output_root: str | Path,
    fraud_type: str,
    output_id: str,
):

    input_path = Path(input_path)
    output_root = Path(output_root)

    # Lire le dossier légitime
    with input_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        dossier = json.load(file)

    # Créer une copie frauduleuse
    fraud_dossier = apply_business_fraud(
        dossier=dossier,
        fraud_type=fraud_type,
    )

    # Nouveau ID
    fraud_dossier["dossier_id"] = output_id

    # Sauvegarde
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
        f"[OK] Fraude BUSINESS créée : {output_path}"
    )

    return output_path


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Génère une fraude Business Rule."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Dossier légitime source",
    )

    parser.add_argument(
        "--output-root",
        default="dossiers_fraude",
    )

    parser.add_argument(
        "--fraud-type",
        required=True,
        choices=[
            "SALAIRE_INVALIDE",
            "ANCIENNETE_INVALIDE",
            "RIB_INVALIDE",
            "IBAN_INVALIDE",
            "DATE_INVALIDE",
        ],
    )

    parser.add_argument(
        "--output-id",
        required=True,
    )

    args = parser.parse_args()

    generate_fraud_business(
        input_path=args.input,
        output_root=args.output_root,
        fraud_type=args.fraud_type,
        output_id=args.output_id,
    )